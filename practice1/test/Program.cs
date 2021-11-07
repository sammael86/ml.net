using System;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

MLContext mlContext = new MLContext(20210916);
const string ModelPath = "d:\\ml\\models\\course\\loan.model";

// Данные
var path = "reviews.tsv";
IDataView dataView = mlContext.Data.LoadFromTextFile<Person>(path, hasHeader: true);

DataOperationsCatalog.TrainTestData trainTestData = mlContext.Data.TrainTestSplit(dataView, 0.2);
IDataView trainData = trainTestData.TrainSet;
IDataView testData = trainTestData.TestSet;

var stopWords = new CustomStopWordsRemovingEstimator.Options
    { StopWords = new[] { "и", "а", "с", "к", "на", "все", "всё" } };

var options = new TextFeaturizingEstimator.Options
{
    StopWordsRemoverOptions = stopWords
};

// Подготовка данных
var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", options,
    nameof(Person.Text));

// Выбираем алгоритм для тренировки
var trainer = mlContext.BinaryClassification.Trainers.SgdCalibrated();
var trainedPipeline = pipeline.Append(trainer);

// Обучение
ITransformer model = trainedPipeline.Fit(trainData);

// Тестирование
var predictions = model.Transform(testData);
var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions);
Console.WriteLine($"Model accuracy: {metrics.Accuracy:P2}");
//
// mlContext.Model.Save(model, trainData.Schema, ModelPath);

// Предсказание
var predEngine = mlContext.Model.CreatePredictionEngine<Person, Prediction>(model);
var p = new Person
{
    Text = "В целом работа выполнена. Но много было просрочек. Не сделано то, что было изначально заложено в проект. Но откликались постоянно и вели работу. Вели работу Если вы насчитываете на «отличную» сдачу, скорее вас ждёт «хорошо» или «средне». Работа выполнена. Особых пререканий нет:)"
};
var prediction = predEngine.Predict(p);
Console.WriteLine($"{prediction.Granted} {prediction.Probability} {prediction.Score}");

class Person
{
    [LoadColumn(0), ColumnName("Label")] public bool IsPositive { get; set; }

    [LoadColumn(1)] public string Text { get; set; }

    [LoadColumn(10)]
    public string CleanText
    {
        get
        {
            var s = Regex.Replace(Text, "[0-9.!]", " ");
            var stringArray = s.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return string.Join(' ', stringArray);
        }
    }
}

public class Prediction
{
    [ColumnName("PredictedLabel")] public bool Granted { get; set; }
    public float Score { get; set; }
    public float Probability { get; set; }
}