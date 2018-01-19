using GalaSoft.MvvmLight;
using NoeliniClassifier.Model;
using System;
using System.Collections.Generic;

namespace NoeliniClassifier.ViewModel
{
    public class NoeliniViewModel : ViewModelBase
    {
        Dictionary<int, Noelini> noelinis;

        public NoeliniViewModel()
        {
            noelinis = new Dictionary<int, Noelini>()
            {
                {0, new Noelini{ Name="Bella", ImagePath=GetPath("Bella") } },
                {1, new Noelini{ Name="Benny", ImagePath=GetPath("Benny") } },
                {2, new Noelini{ Name="Emilie", ImagePath=GetPath("Emilie") } },
                {3, new Noelini{ Name="Flurina", ImagePath=GetPath("Flurina") } },
                {4, new Noelini{ Name="Julie", ImagePath=GetPath("Julie") } },
                {5, new Noelini{ Name="Kira", ImagePath=GetPath("Kira") } },
                {6, new Noelini{ Name="Klaus", ImagePath=GetPath("Klaus") } },
                {7, new Noelini{ Name="Lino", ImagePath=GetPath("Lino") } },
                {8, new Noelini{ Name="Louis", ImagePath=GetPath("Louis") } },
                {9, new Noelini{ Name="Ole", ImagePath=GetPath("Ole") } },
                {10, new Noelini{ Name="Pat", ImagePath=GetPath("Pat") } },
                {11, new Noelini{ Name="Remy", ImagePath=GetPath("Remy") } },
                {12, new Noelini{ Name="Rosa", ImagePath=GetPath("Rosa") } },
                {13, new Noelini{ Name="Stella", ImagePath=GetPath("Stella") } },
                {14, new Noelini{ Name="hmmm...", ImagePath=GetPath("void") } },
            };

            string GetPath(string name) => String.Format("pack://application:,,,/Resources/{0}.png", name);
        }

        private int id;
        public int Id
        {
            get { return id; }
            set
            {
                if (Set(ref id, value))
                {
                    RaisePropertyChanged("Name");
                    RaisePropertyChanged("ImagePath");
                }
            }
        }

        public string Name => noelinis[Id].Name;

        public string ImagePath => noelinis[Id].ImagePath;
    }
}
