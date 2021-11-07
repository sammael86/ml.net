using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(20210928);

var path = "creditcard.csv";
var dataView = mlContext.Data.LoadFromTextFile<CreditCardRecord>(path, ',', true);

var pathTest = "_";
var dataViewTest = mlContext.Data.LoadFromTextFile<CreditCardRecord>(pathTest, ',', true);

// Подготовка данных
var excludedFeatures = new HashSet<string>(new[]
{
    nameof(CreditCardRecord.Time),
    nameof(CreditCardRecord.Class)
});

var featureNames = dataView.Schema
    .Select(x => x.Name)
    .Where(x => !excludedFeatures.Contains(x))
    .ToArray();

var pipeline = mlContext.Transforms.Concatenate("Features", featureNames);

Console.WriteLine("EnsureZeroMean: true");
Train(true);

Console.WriteLine("\nEnsureZeroMean: false");
Train(false);

void Train(bool ensureZeroMean)
{
    for (var i = 1; i < featureNames.Length - 1; i++)
    {
        // Алгоритм тренировки
        var trainer = mlContext.AnomalyDetection.Trainers
            .RandomizedPca(seed: 20210928, rank: i, ensureZeroMean: ensureZeroMean);
        var trainedPipeline = pipeline.Append(trainer);

        // Обучение
        ITransformer model = trainedPipeline.Fit(dataView);

        // Тестирование
        var scoredData = model.Transform(dataViewTest);
        var metrics = mlContext.AnomalyDetection.Evaluate(scoredData, nameof(CreditCardRecord.Class));

        // Предсказание
        var predictions = mlContext.Data
            .CreateEnumerable<CreditCardRecordPrediction>(scoredData, false)
            .Where(x => x.PredictedLabel)
            .ToList();

        Console.WriteLine(
            $"Rank: {i}; model accuracy: {metrics.AreaUnderRocCurve:P2}; found anomalies: {predictions.Count:N0}");
    }
}

internal class CreditCardRecord
{
    [LoadColumn(0)] public float Time { get; set; }
    [LoadColumn(1)] public float V1 { get; set; }
    [LoadColumn(2)] public float V2 { get; set; }
    [LoadColumn(3)] public float V3 { get; set; }
    [LoadColumn(4)] public float V4 { get; set; }
    [LoadColumn(5)] public float V5 { get; set; }
    [LoadColumn(6)] public float V6 { get; set; }
    [LoadColumn(7)] public float V7 { get; set; }
    [LoadColumn(8)] public float V8 { get; set; }
    [LoadColumn(9)] public float V9 { get; set; }
    [LoadColumn(10)] public float V10 { get; set; }
    [LoadColumn(11)] public float V11 { get; set; }
    [LoadColumn(12)] public float V12 { get; set; }
    [LoadColumn(13)] public float V13 { get; set; }
    [LoadColumn(14)] public float V14 { get; set; }
    [LoadColumn(15)] public float V15 { get; set; }
    [LoadColumn(16)] public float V16 { get; set; }
    [LoadColumn(17)] public float V17 { get; set; }
    [LoadColumn(18)] public float V18 { get; set; }
    [LoadColumn(19)] public float V19 { get; set; }
    [LoadColumn(20)] public float V20 { get; set; }
    [LoadColumn(21)] public float V21 { get; set; }
    [LoadColumn(22)] public float V22 { get; set; }
    [LoadColumn(23)] public float V23 { get; set; }
    [LoadColumn(24)] public float V24 { get; set; }
    [LoadColumn(25)] public float V25 { get; set; }
    [LoadColumn(26)] public float V26 { get; set; }
    [LoadColumn(27)] public float V27 { get; set; }
    [LoadColumn(28)] public float V28 { get; set; }
    [LoadColumn(29)] public float Amount { get; set; }
    [LoadColumn(30)] public float Class { get; set; }
}

public class CreditCardRecordPrediction
{
    public bool PredictedLabel { get; set; }
    public float Score { get; set; }
}