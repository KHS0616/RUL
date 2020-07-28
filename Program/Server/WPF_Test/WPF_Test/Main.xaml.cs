using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WPF_Test
{
    public partial class Main : Page
    {
        public string name = "";
        public Main(string name)
        {
            InitializeComponent();
            this.name = name;
        }

        // 센서 간 이동버튼
        private void btn_Sensor_Checked(object sender, RoutedEventArgs e)
        {
            RadioButton btn = sender as RadioButton;
            int findIndex = Array.FindIndex(Server.gateWayName, i => i == name.Split()[0]);
            switch (btn.Content.ToString())
            {
                case "1호기":
                    Server.gateWays[findIndex].thisPage = 1;
                    NavigationService.Navigate(Server.gateWays[findIndex].pages[1]);
                    break;
                case "2호기":
                    Server.gateWays[findIndex].thisPage = 2;
                    NavigationService.Navigate(Server.gateWays[findIndex].pages[2]);
                    break;
                case "3호기":
                    Server.gateWays[findIndex].thisPage = 3;
                    NavigationService.Navigate(Server.gateWays[findIndex].pages[3]);
                    break;
                case "4호기":
                    Server.gateWays[findIndex].thisPage = 4;
                    NavigationService.Navigate(Server.gateWays[findIndex].pages[4]);
                    break;
                default:
                    Server.gateWays[findIndex].thisPage = 0;
                    NavigationService.Navigate(Server.gateWays[findIndex].pages[0]);
                    break;
            }
        }

        // 게이트웨이 및 센서 연결
        private void btn_Connect_Sensor(object sender, RoutedEventArgs e)
        {
            Button btn = sender as Button;
            switch (btn.Content.ToString())
            {
                case "1호기 연결":
                    Mobius.PostProduct("1");
                    Server.mainWindow.AppendText("[보냄]: 계양역 1호기 연결");
                    btn.IsEnabled = false;
                    btn_Disconnect_001.IsEnabled = true;
                    break;
                case "2호기 연결":
                    Mobius.PostProduct("connect sensor002");
                    break;
                case "3호기 연결":
                    Mobius.PostProduct("connect sensor003");
                    break;
                case "4호기 연결":
                    Mobius.PostProduct("connect sensor004");
                    break;
                default:
                    Mobius.PostProduct("connect gateway");
                    break;
            }
        }

        // 게이트웨이 및 센서 연결
        private void btn_Disconnect_Sensor(object sender, RoutedEventArgs e)
        {
            Button btn = sender as Button;
            switch (btn.Content.ToString())
            {
                case "1호기 연결해제":
                    Mobius.PostProduct("3");
                    Server.mainWindow.AppendText("[보냄]: 계양역 1호기 연결 해제");
                    btn.IsEnabled = false;
                    btn_Connect_001.IsEnabled = true;
                    break;
                case "2호기 연결해제":
                    Mobius.PostProduct("connect sensor002");
                    break;
                case "3호기 연결해제":
                    Mobius.PostProduct("connect sensor003");
                    break;
                case "4호기 연결해제":
                    Mobius.PostProduct("connect sensor004");
                    break;
                default:
                    Mobius.PostProduct("connect gateway");
                    break;
            }
        }
    }
}
