using AForge.Video.DirectShow;
using GalaSoft.MvvmLight.Command;
using NoeliniClassifier.ViewModel;
using System;
using System.Drawing;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace NoeliniClassifier.View
{
    public partial class WebcamView : UserControl
    {
        public WebcamView()
        {
            InitializeComponent();
            Dispatcher.ShutdownStarted += DispatcherShutdownStarted;
        }

        private void OnLoaded(object sender, RoutedEventArgs eventArgs)
        {
            var videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);
            VideoPlayer.VideoSource = new VideoCaptureDevice(videoDevices[0].MonikerString);
            VideoPlayer.VideoSource.Start();
        }

        private void OnUnloaded(object sender, RoutedEventArgs eventArgs)
        {
            if (VideoPlayer.VideoSource != null)
            {
                VideoPlayer.VideoSource.SignalToStop();
                VideoPlayer.VideoSource.WaitForStop();
                VideoPlayer.VideoSource.Stop();
                VideoPlayer.VideoSource = null;
            }
        }

        private void DispatcherShutdownStarted(object sender, EventArgs e)
        {
            if (VideoPlayer.VideoSource != null)
            {
                VideoPlayer.VideoSource.SignalToStop();
                VideoPlayer.VideoSource.WaitForStop();
                VideoPlayer.VideoSource.Stop();
                VideoPlayer.VideoSource = null;
            }
        }

        public Bitmap TakeSnapshot()
        {
            try
            {
                var playerPoint = this.VideoPlayer.PointToScreen(new System.Drawing.Point(
                    this.VideoPlayer.ClientRectangle.X, 
                    this.VideoPlayer.ClientRectangle.Y));

                using (var bitmap = new Bitmap(this.VideoPlayer.ClientRectangle.Width, this.VideoPlayer.ClientRectangle.Height))
                {
                    using (var graphicsFromImage = Graphics.FromImage(bitmap))
                    {
                        graphicsFromImage.CopyFromScreen(
                            playerPoint, 
                            System.Drawing.Point.Empty, 
                            new System.Drawing.Size(bitmap.Width, bitmap.Height));
                    }

                    return new Bitmap(bitmap);
                }
            }
            catch (Exception)
            {
                return null;
            }
        }
    }
}