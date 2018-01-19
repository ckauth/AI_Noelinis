using NoeliniClassifier.ViewModel;
using System.Windows;
using System;
using System.Windows.Controls;
using System.ComponentModel;

namespace NoeliniClassifier.View
{
    public partial class NoeliniView : UserControl
    {
        NoeliniViewModel _vm;

        public NoeliniView()
        {
            InitializeComponent();
            _vm = (NoeliniViewModel)this.userControlGrid.DataContext;
        }

        public int NoeliniId
        {
            get { return (int)GetValue(NoeliniIdProperty); }
            set { SetValue(NoeliniIdProperty, value); }
        }

        public static DependencyProperty NoeliniIdProperty =
            DependencyProperty.Register("NoeliniId", typeof(int),
                typeof(NoeliniView), new PropertyMetadata(NoeliniOnIdChanged));

        private static void NoeliniOnIdChanged(
          object sender,
          DependencyPropertyChangedEventArgs args)
        {
            var attachedView = sender as NoeliniView;
            if (attachedView != null)
            {
                attachedView._vm.Id = Int32.Parse(args.NewValue.ToString());
            }
        }
    }
}