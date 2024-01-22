using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using static Microsoft.ML.DataOperationsCatalog;

namespace POC.MLNet
    {
    //input training/test data structure
    public class FeelingInput
        {
        public FeelingInput(string emotin, bool isPositive)
            { 
            Emotion=emotin;
            IsPositive=isPositive;
            }
        public FeelingInput(string emotin)
        {
            Emotion = emotin;
        }
        [LoadColumn(0)]
        public string Emotion { get; set; } 


        [LoadColumn(1)]
        [ColumnName("Label")]//input training data assumptions/result
        public bool IsPositive { get; set; }
        }

    //op predicted results
    public class FeelingOutputPrediction
        {
        [LoadColumn(0)]
        public string Emotion { get; set; } //input


        [LoadColumn(1)]
        [ColumnName("PredictedLabel")]//output final result
        public bool IsPositive { get; set; }
        }
    internal class Program
        {
        static List<FeelingInput> feelingInput = new List<FeelingInput>();
        static void Main(string[] args)
            {
            Console.WriteLine("Hello, World!");

            //Create MLContext
            MLContext mlContext = new MLContext();

            //Load Data
            LoadTrainingData();
            IDataView dataView = mlContext.Data.LoadFromEnumerable(feelingInput);

            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.5);
            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations          
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(FeelingInput.Emotion));//to vectors

            // STEP 3: Set the training algorithm, then create and config the modelBuilder                            
            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            //training the model
            ITransformer trainedModel = trainingPipeline.Fit(trainingData);
            //model is ready now

            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"Accuracy is{metrics.Accuracy*100}%");

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<FeelingInput, FeelingOutputPrediction>(trainedModel);

            // Score
            var inputString = "Its Awesome";
            var resultprediction = predEngine.Predict(new FeelingInput(inputString) { });
            Console.WriteLine(resultprediction.IsPositive);

            Console.ReadLine();
            }

        private static void LoadTrainingData()
            {
            var lst1=new List<FeelingInput>() { new("am good", true), new("am very good", true), new("nice", true), new("good luck", true), new("awesome", true), new("all is well", true) , new("smooth", true), new("excellent", true), new("super", true), new("good", true) };
            var lst2 = new List<FeelingInput>() { new("bad luck", false), new("my fate", false), new("am very bad", false), new("horrific", false), new("terrible", false), new("slow", false), new("bad", false) };
         
            feelingInput.AddRange(lst1);
            feelingInput.AddRange(lst2);
            }
        }
    }
