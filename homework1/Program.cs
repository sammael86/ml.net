using System;
using Microsoft.ML;
using Microsoft.ML.Data;

MLContext mlContext = new(20210920);

// Данные
var testPath = "HeartTest.tsv";
var trainPath = "HeartTraining.tsv";

var trainData = mlContext.Data.LoadFromTextFile<HeartData>(trainPath);
var testData = mlContext.Data.LoadFromTextFile<HeartData>(testPath);

// Подготовка данных
var pipeline = mlContext.Transforms.Concatenate("Features",
    nameof(HeartData.Age),
    nameof(HeartData.Sex),
    nameof(HeartData.ChestPainType),
    nameof(HeartData.RestingBloodPressure),
    nameof(HeartData.SerumCholesterol),
    nameof(HeartData.FastingBloodSugar),
    nameof(HeartData.RestingElectrocardiographyResults),
    nameof(HeartData.MaximumHeartRate),
    nameof(HeartData.ExerciseInducedAngina),
    nameof(HeartData.StDepressionInducedByExercise),
    nameof(HeartData.SlopeOfPeakExerciseStSegment),
    nameof(HeartData.NumberOfMajorVessels),
    nameof(HeartData.Thal));

// Выбираем алгоритм для тренировки
var trainer = mlContext.BinaryClassification.Trainers.SgdNonCalibrated();
var trainedPipeline = pipeline.Append(trainer);

// Обучение
var model = trainedPipeline.Fit(trainData);

// Тестирование
var predictions = model.Transform(testData);
var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions);
Console.WriteLine($"Model accuracy: {metrics.Accuracy:P2}");

// Предсказание
var predEngine = mlContext.Model.CreatePredictionEngine<HeartData, Prediction>(model);
var heartData = new HeartData
{
    Age = 10,
    Sex = 1,
    ChestPainType = 2,
    RestingBloodPressure = 120,
    SerumCholesterol = 190,
    FastingBloodSugar = 0,
    RestingElectrocardiographyResults = 0,
    MaximumHeartRate = 110,
    ExerciseInducedAngina = 1,
    StDepressionInducedByExercise = 0.2f,
    SlopeOfPeakExerciseStSegment = 2,
    NumberOfMajorVessels = 3,
    Thal = 0
};
var prediction = predEngine.Predict(heartData);
Console.WriteLine($"{prediction.DiagnosisOfHeartDisease} {prediction.Score} {prediction.Probability}");

internal class HeartData
{
    [LoadColumn(0)] public float Age { get; set; }
    [LoadColumn(1)] public float Sex { get; set; }
    [LoadColumn(2)] public float ChestPainType { get; set; }
    [LoadColumn(3)] public float RestingBloodPressure { get; set; }
    [LoadColumn(4)] public float SerumCholesterol { get; set; }
    [LoadColumn(5)] public float FastingBloodSugar { get; set; }
    [LoadColumn(6)] public float RestingElectrocardiographyResults { get; set; }
    [LoadColumn(7)] public float MaximumHeartRate { get; set; }
    [LoadColumn(8)] public float ExerciseInducedAngina { get; set; }
    [LoadColumn(9)] public float StDepressionInducedByExercise { get; set; }
    [LoadColumn(10)] public float SlopeOfPeakExerciseStSegment { get; set; }
    [LoadColumn(11)] public float NumberOfMajorVessels { get; set; }
    [LoadColumn(12)] public float Thal { get; set; }
    [LoadColumn(13), ColumnName("Label")] public bool DiagnosisOfHeartDisease { get; set; }
}

public class Prediction
{
    [ColumnName("PredictedLabel")] public bool DiagnosisOfHeartDisease { get; set; }
    public float Score { get; set; }
    public float Probability { get; set; }
}