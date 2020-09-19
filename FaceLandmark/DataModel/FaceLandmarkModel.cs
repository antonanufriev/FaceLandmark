using System;
using System.Collections.Generic;
using System.Text;

namespace FaceLandmark.DataModel
{
    class FaceLandmarkModel: IOnnxModel
    {
        public string ModelPath { get; private set; }

        public string ModelInput { get; } = "input";
        public string ModelOutput { get; } = "output_3";

        public FaceLandmarkModel(string modelPath)
        {
            ModelPath = modelPath;
        }
    }
}
