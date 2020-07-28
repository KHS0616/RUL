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

    public partial class GateWay : Window
    {
        // 각 메뉴, 센서 페이지에 대한 변수
        public Page[] pages = new Page[5];
        public int thisPage = 0;
        public Main mainPage;
        public Sensor[] sensors = new Sensor[4];

        // 게이트웨이 정보
        public string gateWayName = "";

        public GateWay(string namee)
        {
            InitializeComponent();
            // 게이트웨이 정보 저장
            gateWayName = namee;
            // 각 페이지 정보 저장
            mainPage = new Main(string.Format("{0} Main", namee));
            sensors[0] = new Sensor(string.Format("{0} 1호기", namee));
            sensors[1] = new Sensor(string.Format("{0} 2호기", namee));
            sensors[2] = new Sensor(string.Format("{0} 3호기", namee));
            sensors[3] = new Sensor(string.Format("{0} 4호기", namee));

            pages[0] = mainPage;
            pages[1] = sensors[0];
            pages[2] = sensors[1];
            pages[3] = sensors[2];
            pages[4] = sensors[3];

            frame_gateway.Content = pages[0];
        }
    }
}
