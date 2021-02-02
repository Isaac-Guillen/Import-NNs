using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ONNXModelExample
{
    class Program
    {
        static void Main(string[] args)
        {
            string path = System.AppContext.BaseDirectory + "myModel.onnx";

            Console.WriteLine(path);
            Tensor<float> input = new DenseTensor<float>(new[] {32, 32});
            Tensor<float> output = new DenseTensor<float>(new[] {1, 4, 4});
            for (int y = 0; y < 32; y++)
            {
                for (int x = 0; x < 32; x++)
                {
                    input[y, x] = (float)Math.E;
                }
            }

            //Console.WriteLine(input.GetArrayString());

            // Setup inputs
            List<NamedOnnxValue> inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Input", input.Reshape(new [] {1, 32, 32}).ToDenseTensor()),
            };
            // Setup outputs
            List<NamedOnnxValue> outputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("Output", output),
            };

            Stopwatch stopWatch = new Stopwatch();

            stopWatch.Start();

            // Run inference
            InferenceSession session = new InferenceSession(path);
            session.Run(inputs, outputs);
            output = outputs[0].AsTensor<float>();
            Console.WriteLine(output.Reshape(new[] { 4, 4 }).ToDenseTensor().GetArrayString());

            stopWatch.Stop();

            Console.WriteLine(stopWatch.ElapsedMilliseconds.ToString());
        }
    }
}
