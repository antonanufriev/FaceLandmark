using System;
using System.Collections.Generic;
using System.Text;

namespace FaceLandmark.DataModel
{
    interface IOnnxObjectPrediction
    {
        float[] PredictedLabels { get; set; }
    }
}
