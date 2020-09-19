using System;
using System.IO;

namespace FaceLandmark
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start FaceLandmark");
            FaceLanmarker marker = new FaceLanmarker();
            marker.ProcessImage();

        }

        
    }
}
