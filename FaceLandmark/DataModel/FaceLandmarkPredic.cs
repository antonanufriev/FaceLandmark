using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace FaceLandmark.DataModel
{
    class FaceLandmarkPredic : IOnnxObjectPrediction
    {
        float[,,] htMap = new float[68, 64, 64];
        float[] maxIndex_x = new float[68];
        float[] maxIndex_y = new float[68];

        [ColumnName("output_3")]
        public float[] PredictedLabels { get ; set ; }

        public void ParseOutput()
        {
            for (int i=0;i<68; i++)
            {
                float val = 0;
                int maxIndex = 0;
                for (int j = 0; j < 4096; j++)
                {
                    if (val < PredictedLabels[i * 4096 + j])
                    {
                        val = PredictedLabels[i * 4096 + j];
                        maxIndex = j;
                    }
                    int px = j % 64;
                    int py = j / 64;
                    htMap[i, px,py] = PredictedLabels[i * 4096 + j];
                }
                maxIndex += 1;
                maxIndex_x[i] = (float)(maxIndex -1) % 64;
                maxIndex_y[i] = (float)((maxIndex / 64) + 1);
                maxIndex += 1;
            }

            for (int i=0; i<68; i++)
            {
                int pX = (int)maxIndex_x[i];
                int pY = (int)maxIndex_y[i];
                if (pX > 0 && pX< 63 && pY > 0 && pY< 63)
                {
                    float diffX = htMap[i, pX + 1, pY] - htMap[i, pX - 1, pY];
                    float diffY = htMap[i, pX , pY +1] - htMap[i, pX , pY -1];
                    maxIndex_x[i] += MathF.Sign(diffX) * 0.25f;
                    maxIndex_y[i] += MathF.Sign(diffY) * 0.25f;
                    maxIndex_x[i] -= 0.5f;
                    maxIndex_y[i] -= 0.5f;
                }
            }



                

        }

        public void DrawLandmark(Bitmap image)
        {
            float xScale = (image.Width / 256.0f);
            float yScale = (image.Height / 256.0f);

            using (Graphics grf = Graphics.FromImage(image))
            {
                using (Brush brsh = new SolidBrush(ColorTranslator.FromHtml("#ff00ffff")))
                {
                    for (int i = 0; i < 68; i++)
                    {
                        grf.FillEllipse(brsh, (maxIndex_x[i] * 4 * xScale  - 2), (maxIndex_y[i] * 4 * yScale - 2), 4, 4);
                    }
                }
            }
        }
             
    }
}
