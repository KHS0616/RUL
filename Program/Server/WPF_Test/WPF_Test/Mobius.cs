using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;
using System.Windows;

namespace WPF_Test
{
    
    class Mobius
    {
        // 모비우스 접속을 위한 변수 선언
        public static HttpClient client = new HttpClient();

        // REST API를 위한 헤더 변수


        // 모비우스 접속을 위한 클라이언트, REST API 헤더 설정
        public static void CreateClient()
        {
            client.BaseAddress = new Uri("http://192.168.0.10:7579");
            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));
            client.DefaultRequestHeaders.Add("X-M2M-RI", "12345");
            client.DefaultRequestHeaders.Add("X-M2M-Origin", "{{aei}}");
        }

        // Get API 전송
        public static async void GetProductAsync(string path)
        {
            string test="";
            client.BaseAddress = new Uri("http://192.168.0.10:7579");
            client.DefaultRequestHeaders.Accept.Add(
                new MediaTypeWithQualityHeaderValue("application/json"));
            client.DefaultRequestHeaders.Add("X-M2M-RI", "12345");
            client.DefaultRequestHeaders.Add("X-M2M-Origin", "{{aei}}");
            HttpResponseMessage response = await client.GetAsync(path);
            if (response.IsSuccessStatusCode)
            {
                test = await response.Content.ReadAsStringAsync();
            }
            MessageBox.Show(test);
        }

        // Post API 전송
        public static async void PostProduct(string content)
        {
            try
            {
                // 전송할 API Body 생성 및 Content-Type 설정
                var sendAPI = new StringContent("{\n    \"m2m:cin\": {\n        \"con\": \"" + content + "\"\n    }\n}");
                sendAPI.Headers.ContentType = MediaTypeHeaderValue.Parse("application/vnd.onem2m-res+json; ty=4");

                // API 전송 후, 에러 응답이 있을 경우 오류코드를 받는다.
                var response = await client.PostAsync("Hello-Mobius/inhatc-pi/Gateway", sendAPI);
                response.EnsureSuccessStatusCode();
            }
            catch (HttpRequestException ex)
            {
                MessageBox.Show(ex.Message);
            }
            finally
            {
                
            }
        }
    }
}
