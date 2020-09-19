using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;

namespace FaceLandmark.DataModel
{
    public struct ImageSettings
    {
        public const int imageHeight = 256;
        public const int imageWidth = 256;
    }
    public class ImageInputData
    {
        public string ImagePath;
        public string ImageFileName;

        [ImageType(ImageSettings.imageHeight, ImageSettings.imageWidth)]
        public Bitmap Images { get; set; }
        public ImageInputData(string filePath)
        {
            ImagePath = "";//Path.GetDirectoryName(filePath);
            ImageFileName = Path.GetFileName(filePath);
            Images = (Bitmap)Bitmap.FromFile(filePath);
        }

        public static IEnumerable<ImageInputData> ReadFromFile(string imageFolder)
        {
            return Directory
                .GetFiles(imageFolder)
                .Where(filePath => Path.GetExtension(filePath) != ".md")
                .Select(filePath => new ImageInputData(filePath));
        }

    }
}
