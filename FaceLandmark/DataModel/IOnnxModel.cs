using System;
using System.Collections.Generic;
using System.Text;

namespace FaceLandmark.DataModel
{
    interface IOnnxModel
    {
        string ModelPath { get; }

        string ModelInput { get; }
        string ModelOutput { get; }
    }
}
