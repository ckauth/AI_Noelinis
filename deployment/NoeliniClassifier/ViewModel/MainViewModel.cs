/* Uses code from https://github.com/Microsoft/CNTK */

using CNTK;
using GalaSoft.MvvmLight;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Threading;
using System.Xml;

namespace NoeliniClassifier.ViewModel
{
    public class MainViewModel : ViewModelBase
    {
        DispatcherTimer dispatcherTimer;

        private int confidenceLevel;
        public int ConfidenceLevel
        {
            get => confidenceLevel;
            set { Set(ref confidenceLevel, value); }
        }

        public delegate Bitmap TakeSnapshotDelegate();
        public TakeSnapshotDelegate takeSnapshot;

        DeviceDescriptor device;
        Function modelFunc;
        List<float> meanImage;

        public MainViewModel()
        {
            LoadModel();

            NoeliniId = 14;
            ConfidenceLevel = 0;
            
            dispatcherTimer = new DispatcherTimer();
            dispatcherTimer.Tick += new EventHandler(dispatcherTimer_Tick);
            dispatcherTimer.Interval = new TimeSpan(0, 0, 1);
            dispatcherTimer.Start();
        }

        private int noeliniId;
        public int NoeliniId
        {
            get => noeliniId;
            set
            {
                if (ConfidenceLevel > 50)
                    Set(ref noeliniId, value);
                else
                    Set(ref noeliniId, 14);
            }
        }

        private void dispatcherTimer_Tick(object sender, EventArgs e)
        {
            var snapshot = takeSnapshot.Invoke();

            snapshot = CropBitmap(
                snapshot,
                (snapshot.Width - snapshot.Height) / 2,
                0,
                snapshot.Height,
                snapshot.Height);

            var inputShape = modelFunc.Arguments.Single().Shape;
            snapshot = ResizeBitmap(
                snapshot,
                inputShape[0],
                inputShape[1]);

            Thumbnail = ConvertToImageSource(snapshot);

            GetProbabilities(snapshot);
        }

        private ImageSource thumbnail;
        public ImageSource Thumbnail
        {
            get => thumbnail;
            set { Set(ref thumbnail, value); }
        }

        private void LoadModel()
        {
            // set the device
            device = DeviceDescriptor.CPUDevice;

            // load the model
            var assembly = Assembly.GetExecutingAssembly();
            string modelResource = "NoeliniClassifier.Resources.NoeliniModel.dnn";
            using (Stream resFilestream = assembly.GetManifestResourceStream(modelResource))
            {
                byte[] modelBuffer = new byte[resFilestream.Length];
                resFilestream.Read(modelBuffer, 0, modelBuffer.Length);
                modelFunc = Function.Load(modelBuffer, device);
            }

            // load the mean image
            string meanImageResource = "NoeliniClassifier.Resources.mean_image.xml";         
            using (Stream stream = assembly.GetManifestResourceStream(meanImageResource))
            using (StreamReader reader = new StreamReader(stream))
            {
                string result = reader.ReadToEnd();
                XmlDocument meanImageDoc = new XmlDocument();
                meanImageDoc.LoadXml(result);
                var meanImageAsString = meanImageDoc.DocumentElement.LastChild.LastChild.InnerText;
                meanImage = meanImageAsString.Split(' ').Select(Convert.ToSingle).ToList();
            }
        }

        private Bitmap CropBitmap(Bitmap bitmap, int cropX, int cropY, int cropWidth, int cropHeight)
        {
            Rectangle rect = new Rectangle(cropX, cropY, cropWidth, cropHeight);
            Bitmap cropped = bitmap.Clone(rect, bitmap.PixelFormat);
            return cropped;
        }
        
        public Bitmap ResizeBitmap(Bitmap bitmap, int width, int height)
        {
            var resized = new Bitmap(width, height);
            resized.SetResolution(bitmap.HorizontalResolution, bitmap.VerticalResolution);

            using (var g = Graphics.FromImage(resized))
            {
                g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Default;
                g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.Default;
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.Default;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Default;

                var attributes = new ImageAttributes();
                attributes.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                g.DrawImage(
                    bitmap, 
                    new Rectangle(0, 0, width, height), 
                    0, 
                    0, 
                    bitmap.Width,
                    bitmap.Height, 
                    GraphicsUnit.Pixel, 
                    attributes);
            }

            return resized;
        }

        public ImageSource ConvertToImageSource(Bitmap bitmap)
        {
            var imageSourceConverter = new ImageSourceConverter();
            using (var memoryStream = new MemoryStream())
            {
                bitmap.Save(memoryStream, ImageFormat.Png);
                var snapshotBytes = memoryStream.ToArray();
                return (ImageSource)imageSourceConverter.ConvertFrom(snapshotBytes); ;
            }
        }

        public void GetProbabilities(Bitmap bitmap)
        {
            try
            {
                Variable inputVar = modelFunc.Arguments.Single();
                NDShape inputShape = inputVar.Shape;

                List<float> pictureCHW = ParallelExtractCHW(bitmap);
                List<float> resizedCHW = new List<float>();

                using (var e1 = pictureCHW.GetEnumerator())
                using (var e2 = meanImage.GetEnumerator())
                {
                    while (e1.MoveNext() && e2.MoveNext())
                    {
                        resizedCHW.Add(e1.Current - float.Parse(e2.Current.ToString()));
                    }
                }

                // Create input data map
                var inputDataMap = new Dictionary<Variable, Value>();
                var inputVal = Value.CreateBatch(inputShape, resizedCHW, device);
                inputDataMap.Add(inputVar, inputVal);

                // The model has only one output.
                // You can also use the following way to get output variable by name:
                // Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();
                Variable outputVar = modelFunc.Output;

                // Create output data map. Using null as Value to indicate using system allocated memory.
                // Alternatively, create a Value object and add it to the data map.
                var outputDataMap = new Dictionary<Variable, Value>();
                outputDataMap.Add(outputVar, null);

                // Start evaluation on the device
                modelFunc.Evaluate(inputDataMap, outputDataMap, device);

                // Get evaluate result as dense output
                var outputVal = outputDataMap[outputVar];
                var probabilities = outputVal.GetDenseData<float>(outputVar)[0];

                ConfidenceLevel = (int)Math.Round(probabilities.Max() * 100);
                NoeliniId = probabilities.IndexOf(probabilities.Max());

                PrintOutput(outputVar.Shape.TotalSize, outputVal.GetDenseData<float>(outputVar));
            }

            catch (Exception ex)
            {
                Console.WriteLine("Error: {0}\nCallStack: {1}\n Inner Exception: {2}", ex.Message, ex.StackTrace, ex.InnerException != null ? ex.InnerException.Message : "No Inner Exception");
                throw ex;
            }
        }

        internal void PrintOutput<T>(int sampleSize, IList<IList<T>> outputBuffer)
        {
            Console.WriteLine("The number of sequences in the batch: " + outputBuffer.Count);
            int seqNo = 0;
            int outputSampleSize = sampleSize;
            foreach (var seq in outputBuffer)
            {
                if (seq.Count % outputSampleSize != 0)
                {
                    throw new ApplicationException("The number of elements in the sequence is not a multiple of sample size");
                }

                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count / outputSampleSize));
                int i = 0;
                int sampleNo = 0;
                foreach (var element in seq)
                {
                    if (i++ % outputSampleSize == 0)
                    {
                        Console.Write(String.Format("    sample {0}: ", sampleNo));
                    }
                    Console.Write(element);
                    if (i % outputSampleSize == 0)
                    {
                        Console.WriteLine(".");
                        sampleNo++;
                    }
                    else
                    {
                        Console.Write(",");
                    }
                }
            }
        }

        /// <summary>
        /// Extracts image pixels in CHW using parallelization
        /// </summary>
        /// <param name="image">The bitmap image to extract features from</param>
        /// <returns>A list of pixels in CHW order</returns>
        public static List<float> ParallelExtractCHW(Bitmap image)
        {
            int channelStride = image.Width * image.Height;
            int imageWidth = image.Width;
            int imageHeight = image.Height;

            var features = new byte[imageWidth * imageHeight * 3];
            var bitmapData = image.LockBits(new System.Drawing.Rectangle(0, 0, imageWidth, imageHeight), ImageLockMode.ReadOnly, image.PixelFormat);
            IntPtr ptr = bitmapData.Scan0;
            int bytes = Math.Abs(bitmapData.Stride) * bitmapData.Height;
            byte[] rgbValues = new byte[bytes];

            int stride = bitmapData.Stride;

            // Copy the RGB values into the array.
            System.Runtime.InteropServices.Marshal.Copy(ptr, rgbValues, 0, bytes);

            // The mapping depends on the pixel format
            // The mapPixel lambda will return the right color channel for the desired pixel
            Func<int, int, int, int> mapPixel = GetPixelMapper(image.PixelFormat, stride);

            Parallel.For(0, imageHeight, (int h) =>
            {
                Parallel.For(0, imageWidth, (int w) =>
                {
                    Parallel.For(0, 3, (int c) =>
                    {
                        features[channelStride * c + imageWidth * h + w] = rgbValues[mapPixel(h, w, c)];
                    });
                });
            });

            image.UnlockBits(bitmapData);

            return features.Select(b => (float)b).ToList();
        }

        /// <summary>
        /// Returns a function for extracting the R-G-B values properly from an image based on its pixel format
        /// </summary>
        /// <param name="pixelFormat">The image's pixel format</param>
        /// <param name="heightStride">The stride (row byte count)</param>
        /// <returns>A function with signature (height, width, channel) returning the corresponding color value</returns>
        private static Func<int, int, int, int> GetPixelMapper(System.Drawing.Imaging.PixelFormat pixelFormat, int heightStride)
        {
            switch (pixelFormat)
            {
                case System.Drawing.Imaging.PixelFormat.Format32bppArgb:
                    return (h, w, c) => h * heightStride + w * 4 + c;  // bytes are B-G-R-A
                case System.Drawing.Imaging.PixelFormat.Format24bppRgb:
                default:
                    return (h, w, c) => h * heightStride + w * 3 + c;  // bytes are B-G-R
            }
        }
    }
}