using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
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
using System.Windows.Threading;

namespace RUL_Program
{
    public partial class GatewaySensorPage : Page
    {
        // 스펙 정보
        public string Step_Fs = "";
        public string Step_Time = "";
        public string Step_DifStep = "";
        public string Step_PerSpec = "";

        public string RS_Fs = "";
        public string RS_Time = "";
        public string RS_DifLink = "";

        public GatewaySensorPage(string name)
        {
            InitializeComponent();
            lblSensorName.Content = name;
        }

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
                    imgResult.Source = bitmap;
                }));
            }
        }
    }
}
