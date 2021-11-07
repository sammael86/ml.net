using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using NumSharp.Extensions;

// using Microsoft.ML.AutoML;

MLContext mlContext = new(20210923);

// Данные
var path = "bikes.csv";
var dailyRecordInfos = File.ReadLines(path)
    .Skip(1)
    .Select(MapToDailyRecord)
    .ToArray();

var rowsDataView = mlContext.Data.LoadFromEnumerable(dailyRecordInfos);
var shuffledData = mlContext.Data.ShuffleRows(rowsDataView, 20210923);
var trainTestData = mlContext.Data.TrainTestSplit(shuffledData, 0.2);
var trainData = trainTestData.TrainSet;
var testData = trainTestData.TestSet;

// Подготовка данных
var excludedFeatures = new HashSet<string>(new[]
{
    string.Empty, 
    // nameof(DailyRecord.Number),
    // nameof(DailyRecord.Date),
    // nameof(DailyRecord.Season),
    // nameof(DailyRecord.Year),
    // nameof(DailyRecord.Month),
    // nameof(DailyRecord.IsHoliday),
    // nameof(DailyRecord.DayOfWeek),
    // nameof(DailyRecord.IsWorkingDay),
    // nameof(DailyRecord.Weather),
    // nameof(DailyRecord.Temperature),
    // nameof(DailyRecord.TemperatureFeelsLike),
    // nameof(DailyRecord.Humidity),
    // nameof(DailyRecord.Wind),
    // nameof(DailyRecord.Guests),
    // nameof(DailyRecord.Regular),
});

var featureNames = rowsDataView.Schema
    .Select(x => x.Name)
    .Where(x => !excludedFeatures.Contains(x))
    .ToArray();

var dropColumns = rowsDataView.Schema.Select(x => x.Name)
    .Where(x => x != "Label" && !featureNames.Contains(x))
    .ToArray();

var pipeline = mlContext.Transforms.Concatenate("Features", featureNames)
    .Append(mlContext.Transforms.NormalizeLogMeanVariance("Features"));


var dataPreview = pipeline.Preview(trainData, dailyRecordInfos.Length, dailyRecordInfos.Length);
List<DailyRecord> denormalizedFeatures = new();
List<DailyRecord> normalizedFeatures = new();
foreach (var rowInfo in dataPreview.RowView)
{
    var switcher = true;
    foreach (var vector in rowInfo.Values[^1..])
    {

        var tt = rowInfo.Values[^1];
        if (switcher)
        {
            if (vector.Value is VBuffer<float> vBuffer)
            {
                var dailyInfos = vBuffer.DenseValues().ToList();
                var dailyRecord = MapEnumerableToDailyRecord(dailyInfos);
                denormalizedFeatures.Add(dailyRecord);
            }
            switcher = false;
        }
        else
        {
            if (vector.Value is Microsoft.ML.Data.VBuffer<float> vBuffer)
            {
                var dailyInfos = vBuffer.DenseValues().ToList();
                var dailyRecord = MapEnumerableToDailyRecord(dailyInfos);
                normalizedFeatures.Add(dailyRecord);
            }
            switcher = true;
        }
    }
}


;
// mlContext.Data.CreateEnumerable<DailyRecord>(prev, false);

// var dv = mlContext.Data.Loadfrom

// Выбираем алгоритм для тренировки
// var experimentResult = mlContext.Auto().CreateRegressionExperiment(30)
//     .Execute(trainData);
// var bestRun = experimentResult.BestRun;
// var model = bestRun.Model;

var trainer = mlContext.Regression.Trainers.Sdca();
var trainedPipeline = pipeline.Append(trainer); //bestRun.Estimator);

var model2 = trainedPipeline.Fit(trainData);

// Тестирование
var crossValidationResults = mlContext.Regression.CrossValidate(trainData, trainedPipeline);
Console.WriteLine($"Cross-validated accuracy: {crossValidationResults.Average(cv => cv.Metrics.RSquared):P5}");

var predictions = model2.Transform(testData);
var metrics = mlContext.Regression.Evaluate(predictions);
Console.WriteLine($"Model accuracy: {metrics.RSquared:P5}");

// Предсказание
var predEngine = mlContext.Model.CreatePredictionEngine<DailyRecord, Prediction>(model2);
var sample = MapToDailyRecord("178,6/27/2011,3,0,6,0,1,1,2,0.6825,0.637004,0.658333,0.107588,854,3854,0");
var prediction = predEngine.Predict(sample);
Console.WriteLine($"{prediction.Score}");

PeekDataViewInConsole(trainData, trainedPipeline, 5);

void PeekDataViewInConsole(IDataView dataView, IEstimator<ITransformer> estimator, int numberOfRows = 4)
{
    string msg = $"Peek data in DataView: Showing {numberOfRows} rows with the columns";
    ConsoleWriteHeader(msg);

    var transformer = estimator.Fit(dataView);
    var transformedData = transformer.Transform(dataView);

    var preViewTransformedData = transformedData.Preview(maxRows: numberOfRows);

    var defaultColor = Console.ForegroundColor;
    foreach (var row in preViewTransformedData.RowView)
    {
        var columnCollection = row.Values;
        string lineToPrint = "Row--> ";
        foreach (KeyValuePair<string, object> column in columnCollection)
        {
            if (column.Key == "Label" || column.Key == "Score")
            {
                Console.Write($"{lineToPrint} | ");
                Console.ForegroundColor = ConsoleColor.Red;
                Console.Write($"{column.Key}:{column.Value}");
                Console.ForegroundColor = defaultColor;
                lineToPrint = "";
            }
            else
            {
                lineToPrint += $" | {column.Key}:{column.Value}";
            }
        }

        Console.WriteLine(lineToPrint);
        Console.WriteLine();
    }
}

void ConsoleWriteHeader(params string[] lines)
{
    var defaultColor = Console.ForegroundColor;
    Console.ForegroundColor = ConsoleColor.Yellow;
    Console.WriteLine(" ");
    foreach (var line in lines)
    {
        Console.WriteLine(line);
    }

    var maxLength = lines.Select(x => x.Length).Max();
    Console.WriteLine(new string('#', maxLength));
    Console.ForegroundColor = defaultColor;
}

DailyRecord MapEnumerableToDailyRecord(List<float> dailyInfos)
{
    return new DailyRecord()
    {
        Number = dailyInfos[0],
        Date = dailyInfos[1],
        Season = dailyInfos[2],
        Year = dailyInfos[3],
        Month = dailyInfos[4],
        IsHoliday = dailyInfos[5],
        DayOfWeek = dailyInfos[6],
        IsWorkingDay = dailyInfos[7],
        Weather = dailyInfos[8],
        Temperature = dailyInfos[9],
        TemperatureFeelsLike = dailyInfos[10],
        Humidity = dailyInfos[11],
        Wind = dailyInfos[12],
        Guests = dailyInfos[13],
        Regular = dailyInfos[14],
        Total = dailyInfos[15],
    };
}

DailyRecord MapToDailyRecord(string dailyInfoString)
{
    var dailyRecordInfo = dailyInfoString.Split(',', StringSplitOptions.TrimEntries);
    DateTime.TryParse(dailyRecordInfo[1], out var dateTime);
    var date = dateTime.ToOADate();
    return new DailyRecord()
    {
        Number = float.Parse(dailyRecordInfo[0]),
        Date = (float)date,
        Season = float.Parse(dailyRecordInfo[2]),
        Year = float.Parse(dailyRecordInfo[3]),
        Month = float.Parse(dailyRecordInfo[4]),
        IsHoliday = float.Parse(dailyRecordInfo[5]),
        DayOfWeek = float.Parse(dailyRecordInfo[6]),
        IsWorkingDay = float.Parse(dailyRecordInfo[7]),
        Weather = float.Parse(dailyRecordInfo[8]),
        Temperature = float.Parse(dailyRecordInfo[9]),
        TemperatureFeelsLike = float.Parse(dailyRecordInfo[10]),
        Humidity = float.Parse(dailyRecordInfo[11]),
        Wind = float.Parse(dailyRecordInfo[12]),
        Guests = float.Parse(dailyRecordInfo[13]),
        Regular = float.Parse(dailyRecordInfo[14]),
        Total = float.Parse(dailyRecordInfo[15]),
    };
}

internal class DailyRecord
{
    [LoadColumn(0)] public float Number { get; set; }
    [LoadColumn(1)] public float Date { get; set; }
    [LoadColumn(2)] public float Season { get; set; }
    [LoadColumn(3)] public float Year { get; set; }
    [LoadColumn(4)] public float Month { get; set; }
    [LoadColumn(5)] public float IsHoliday { get; set; }
    [LoadColumn(6)] public float DayOfWeek { get; set; }
    [LoadColumn(7)] public float IsWorkingDay { get; set; }
    [LoadColumn(8)] public float Weather { get; set; }
    [LoadColumn(9)] public float Temperature { get; set; }
    [LoadColumn(10)] public float TemperatureFeelsLike { get; set; }
    [LoadColumn(11)] public float Humidity { get; set; }
    [LoadColumn(12)] public float Wind { get; set; }
    [LoadColumn(13)] public float Guests { get; set; }
    [LoadColumn(14)] public float Regular { get; set; }
    [LoadColumn(15)] [ColumnName("Label")] public float Total { get; set; }
}

internal class Prediction
{
    public float Score { get; set; }
}