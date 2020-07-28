using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Security.Policy;
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
using System.Windows.Threading;

namespace WPF_Test
{
    public partial class Sensor : Page
    {
        // JSON 변수 선언
        JObject content = new JObject();

        // 스펙 정보
        public string Step_Fs = "";
        public string Step_Time = "";
        public string Step_DifStep = "";
        public string Step_PerSpec = "";

        public string RS_Fs = "";
        public string RS_Time = "";
        public string RS_DifLink = "";

        public Sensor(string name)
        {
            InitializeComponent();

            // 라벨 이름 설정
            lbl_name.Content = name;

            // JSON 초기화
            content.Add("Type", "");
            content.Add("Name", "");
            content.Add("Content", "");
            content.Add("Data", "");
            content.Add("Time", "");
            content.Add("Comment", "");

            // 신율을 계산하기 이전 버튼 전부 비활성화
            btn_Bandpass.IsEnabled = false;
            btn_Peak.IsEnabled = false;
            btn_Raw.IsEnabled = false;
        }

        // 스펙 정보 초기화 메소드
        private void InitializeSetting()
        {
            content["Type"] = "Step";
            content["Name"] = lbl_name.Content.ToString();
            content["Content"] = "Load Spec Setting";
            content["Data"] = "";
            content["Time"] = System.DateTime.Now.ToString();
            content["Comment"] = "";
            Server.SendText(content.ToString());
        }

        // 센서 간 이동버튼
        private async void btn_Sensor_Checked(object sender, RoutedEventArgs e)
        {
            int findIndex = Array.FindIndex(Server.gateWayName, i => i == lbl_name.Content.ToString().Split()[0]);
            Button btn = sender as Button;
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
                case "스펙 설정":
                    // 스펙 정보 초기화
                    InitializeSetting();
                    if (lbl_Type.Content.ToString().Equals("Step"))
                    {
                        await Task.Delay(2000);
                        Window Setting_Window = new Setting_Step(lbl_name.Content.ToString());
                        Setting_Window.Show();                        
                    }
                    else
                    {

                    }
                    break;
                default:
                    Server.gateWays[findIndex].thisPage = 0;
                    NavigationService.Navigate(Server.gateWays[findIndex].pages[0]);
                    break;
            }
        }

        // Json 명령 전송 이벤트
        private void btn_SendJson(object sender, RoutedEventArgs e)
        {
            Button btn = sender as Button;
            switch (btn.Content.ToString())
            {
                case "신율 측정":
                    content["Type"] = lbl_Type.Content.ToString();
                    content["Name"] = lbl_name.Content.ToString();
                    content["Content"] = "Sin";
                    content["Data"] = "";
                    content["Time"] = System.DateTime.Now.ToString();
                    content["Comment"] = "";
                    Server.SendText(content.ToString());

                    btn_Sin.IsEnabled = false;
                    btn_Bandpass.IsEnabled = true;
                    btn_Peak.IsEnabled = true;
                    btn_Raw.IsEnabled = true;
                    break;
                case "Raw Graph":
                    content["Type"] = lbl_Type.Content.ToString();
                    content["Name"] = lbl_name.Content.ToString();
                    content["Content"] = "Raw";
                    content["Data"] = "";
                    content["Time"] = System.DateTime.Now.ToString();
                    content["Comment"] = "";
                    Server.SendText(content.ToString());
                    break;
                case "Bandpass Graph":
                    content["Type"] = lbl_Type.Content.ToString();
                    content["Name"] = lbl_name.Content.ToString();
                    content["Content"] = "Bandpass";
                    content["Data"] = "";
                    content["Time"] = System.DateTime.Now.ToString();
                    content["Comment"] = "";
                    Server.SendText(content.ToString());
                    break;
                case "Find Peak Graph":
                    content["Type"] = lbl_Type.Content.ToString();
                    content["Name"] = lbl_name.Content.ToString();
                    content["Content"] = "Peak";
                    content["Data"] = "";
                    content["Time"] = System.DateTime.Now.ToString();
                    content["Comment"] = "";
                    Server.SendText(content.ToString());
                    break;
                default:
                    break;
            }
        }

        // 스텝, 구동 구분
        private void btn_Change_Type(object sender, RoutedEventArgs e)
        {
            Button btn = sender as Button;
            switch (btn.Content.ToString())
            {
                case "Step":
                    lbl_Type.Content = "Step";
                    break;
                case "RS":
                    lbl_Type.Content = "RS";
                    break;
                default:
                    break;
            }
        }

        // 이미지 저장 및 변경 메소드
        public void byteArrayToImage(string msg)
        {
            byte[] msgByte = Convert.FromBase64String(msg);
            TypeConverter tc = TypeDescriptor.GetConverter(typeof(Bitmap));
            using (Bitmap b = (Bitmap)tc.ConvertFrom(msgByte))
            {
                // 외부 스레드에서 컨트롤 접근시 발생하는 에러 방지
                Dispatcher.Invoke(DispatcherPriority.Normal, new Action(delegate
                {
                    // 상대 경로 설정
                    string strPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName;
                    strPath = strPath + @"\Resources\image001.PNG";
                    // 이미지 저장
                    b.Save(strPath, System.Drawing.Imaging.ImageFormat.Png);

                    // 저장된 이미지를 불러와서 출력
                    // 메모리 해제 과정 포함
                    var bitmap = new BitmapImage();
                    var stream = File.OpenRead(strPath);
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.StreamSource = stream;
                    bitmap.EndInit();
                    stream.Close();
                    stream.Dispose();

                    // 이미지 출력
                    // 비트맵 초기화 방지
                    bitmap.Freeze();
                    img_result.Source = bitmap;
                }));
            }
        }
    }
}
