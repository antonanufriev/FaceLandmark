using FaceLandmark.DataModel;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Image;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;

namespace FaceLandmark
{
    class ModelConfigurator
    {
        private readonly MLContext mlContext;
        private readonly ITransformer mlModel;

        public ModelConfigurator(IOnnxModel onnxModel)
        {
            mlContext = new MLContext();
            // Model creation and pipeline definition for images needs to run just once,
            // so calling it from the constructor:
            mlModel = SetupMlNetModel(onnxModel);
        }

        private ITransformer SetupMlNetModel(IOnnxModel onnxModel)
        {
            //ImageInputData imageData = new ImageInputData("f:\\Python\\Deep Learning\\face_alignment_onnx\\images\\2.jpg");
            var dataView = mlContext.Data.LoadFromEnumerable(new List<ImageInputData>());
            var pipeline = mlContext.Transforms
                .ResizeImages(
                    resizing: ImageResizingEstimator.ResizingKind.Fill,
                    outputColumnName: "ResizeOut",
                    imageWidth: ImageSettings.imageWidth,
                    imageHeight: ImageSettings.imageHeight,
                    inputColumnName: nameof(ImageInputData.Images))
                .Append(mlContext.Transforms.ExtractPixels(
                    inputColumnName: "ResizeOut",
                    outputColumnName: "input",
                    interleavePixelColors: false,
                    colorsToExtract: Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorBits.Rgb,
                    orderOfExtraction: Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator.ColorsOrder.ABGR,
                    scaleImage: 1.0f / 255.0f))
                /*
                .Append(mlContext.Transforms.Concatenate(
                    outputColumnName: "input",
                    inputColumnNames: new string[] { "ExtractOut", "ExtractOut" }
                    ));*/
                .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: onnxModel.ModelPath, outputColumnName: onnxModel.ModelOutput, inputColumnName: onnxModel.ModelInput));
            var mlNetModel = pipeline.Fit(dataView);

            /*
            List<ImageInputData> list = new List<ImageInputData>();
            list.Add(imageData);
            dataView = mlContext.Data.LoadFromEnumerable(list);

            var transformedData = pipeline.Fit(dataView).Transform(dataView);
            PrintColumns(transformedData);
            */

            //DataDebuggerPreview debugView = DebuggerExtensions.Preview(pipeline, dataView);
            return mlNetModel;
        }

        public PredictionEngine<ImageInputData, FaceLandmarkPredic> GetMlNetPredictionEngine()
        {
            return mlContext.Model.CreatePredictionEngine<ImageInputData, FaceLandmarkPredic>(mlModel);
        }

        private static void PrintColumns(IDataView transformedData)
        {
            Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25} {4, -25}",
                "ImagePath", "Name", "ImageObject", "ImageObjectResized", "Pixels");

            using (var cursor = transformedData.GetRowCursor(transformedData
                .Schema))
            {
                // Note that it is best to get the getters and values *before*
                // iteration, so as to faciliate buffer sharing (if applicable), and
                // column -type validation once, rather than many times.

                ReadOnlyMemory<char> imagePath = default;
                ReadOnlyMemory<char> name = default;
                Bitmap imageObject = null;
                Bitmap resizedImageObject = null;
                VBuffer<float> pixels = default;

                var imagePathGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["ImagePath"]);

                var nameGetter = cursor.GetGetter<ReadOnlyMemory<char>>(cursor
                    .Schema["ImageFileName"]);

                var imageObjectGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                    "Images"]);

                var resizedImageGetter = cursor.GetGetter<Bitmap>(cursor.Schema[
                    "ResizeOut"]);

                var pixelsGetter = cursor.GetGetter<VBuffer<float>>(cursor.Schema[
                    "ExtractOut"]);

                while (cursor.MoveNext())
                {

                    imagePathGetter(ref imagePath);
                    nameGetter(ref name);
                    imageObjectGetter(ref imageObject);
                    resizedImageGetter(ref resizedImageObject);
                    pixelsGetter(ref pixels);

                    Console.WriteLine("{0, -25} {1, -25} {2, -25} {3, -25} " +
                        "{4, -25}", imagePath, name, imageObject.PhysicalDimension,
                        resizedImageObject.PhysicalDimension, string.Join(",",
                        pixels.DenseValues().Take(30)) + "...");
                }

                // Dispose the image.
                imageObject.Dispose();
                resizedImageObject.Dispose();
            }
        }
    }
}
