using System.Windows;

using NoeliniClassifier.ViewModel;
using static NoeliniClassifier.ViewModel.MainViewModel;

namespace NoeliniClassifier.View
{
    public partial class MainWindow : Window
    {
        MainViewModel _vm;

        public MainWindow()
        {
            InitializeComponent();
            _vm = (MainViewModel)this.DataContext;
            _vm.takeSnapshot = new TakeSnapshotDelegate(WebcamStream.TakeSnapshot);
        }
    }
}
