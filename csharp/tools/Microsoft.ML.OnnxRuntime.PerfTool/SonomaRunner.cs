using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Scoring;
using System.Diagnostics;

namespace Microsoft.ML.OnnxRuntime.PerfTool
{
    public class SonomaRunner
    {
        public static void RunModelSonoma(string modelPath, string inputPath, int iteration, DateTime[] timestamps)
        {
            if (timestamps.Length != (int)TimingPoint.TotalCount)
            {
                throw new ArgumentException("Timestamps array must have " + (int)TimingPoint.TotalCount + " size");
            }

            timestamps[(int)TimingPoint.Start] = DateTime.Now;

            var modelName = "MyModel";
            using (var modelManager = new ModelManager(modelPath, true))
            {
                modelManager.InitOnnxModel(modelName, int.MaxValue);
                timestamps[(int)TimingPoint.ModelLoaded] = DateTime.Now;

                //var inputType = modelManager.GetInputTypeDict("MyModel", int.MaxValue);

                var inputShapes = modelManager.GetInputShapesDict("MyModel", int.MaxValue);
                Tensor[] inputs = new Tensor[inputShapes.Count];
                string[] inputNames = new string[inputShapes.Count];
                inputShapes.Keys.CopyTo(inputNames, 0);
                var outputTypes = modelManager.GetOutputTypeDict("MyModel", int.MaxValue);
                string[] outputNames = new string[outputTypes.Keys.Count];
                outputTypes.Keys.CopyTo(outputNames, 0);

                int index = 0;
                foreach (var name in inputShapes.Keys)
                {
                    long[] shape = inputShapes[name];
                    float[] inputData0 = Program.LoadTensorFromFile(inputPath);
                    inputs[index++] = Tensor.Create(inputData0, shape);
                }

                timestamps[(int)TimingPoint.InputLoaded] = DateTime.Now;

                for (int i = 0; i < iteration; i++)
                {
                    var outputs = modelManager.RunModel(
                                                    modelName,
                                                    int.MaxValue,
                                                    inputNames,
                                                    inputs,
                                                    outputNames
                                                    );
                    Debug.Assert(outputs != null);
                    Debug.Assert(outputs.Length == 1);
                }

                timestamps[(int)TimingPoint.RunComplete] = DateTime.Now;
            }
        }
    }
}
