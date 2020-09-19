using FaceLandmark.DataModel;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.IO;
using System.Text;

namespace FaceLandmark
{
    class FaceLanmarker
    {
        public string assetsPath;
        public string modelFilePath;
        public string imagesFolder;
        public string outputFolder;
        private PredictionEngine<ImageInputData, FaceLandmarkPredic> faceLandmarkPredictionEngine;
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

        public FaceLanmarker()
        {
            assetsPath = GetAbsolutePath(".");
            modelFilePath = Path.Combine(assetsPath, "Model", "model.onnx");
            imagesFolder = Path.Combine(assetsPath, "images");
            outputFolder = Path.Combine(assetsPath, "output");

            FaceLandmarkModel model = new FaceLandmarkModel(modelFilePath);
            ModelConfigurator modelConf = new ModelConfigurator(model);
            faceLandmarkPredictionEngine = modelConf.GetMlNetPredictionEngine();

        }

        public void ProcessImage()
        {
            IEnumerable<ImageInputData> images = ImageInputData.ReadFromFile(imagesFolder);
            using (IEnumerator<ImageInputData> imageDataIter = images.GetEnumerator())
            {
                while (imageDataIter.MoveNext())
                {
                    FaceLandmarkPredic predic = new FaceLandmarkPredic();
                    ImageInputData imageData = imageDataIter.Current;
                    faceLandmarkPredictionEngine.Predict(imageData, ref predic);
                    predic.ParseOutput();
                    predic.DrawLandmark(imageData.Images);
                    FileStream file = new FileStream(outputFolder +"\\"+ imageData.ImageFileName, FileMode.Create);
                    imageData.Images.Save(file, ImageFormat.Jpeg);
                    file.Close();
                    Console.WriteLine(imageData.ImageFileName + "---" + predic.PredictedLabels.Length);
                }
            }
        }



        public void LoadModel()
        {
            //var data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());
        }
    }
}
