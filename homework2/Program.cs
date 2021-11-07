using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

MLContext mlContext = new(20210922);

// Данные
Dictionary<string, int> carModels = new();
Dictionary<string, int> owners = new();
Dictionary<string, int> fuelTypes = new();
Dictionary<string, int> transmissions = new();

var path = "cars.csv";
var carInfos = File.ReadLines(path)
    .Skip(1)
    .Select(MapToCarInfo)
    .ToArray();

var pathOut = "cars_out.csv";
File.WriteAllLines(pathOut, carInfos.Select(ci => ci.ToString()));

var dataView = mlContext.Data.LoadFromEnumerable(carInfos);
var shuffledData = mlContext.Data.ShuffleRows(dataView, 20210922);
var trainTestData = mlContext.Data.TrainTestSplit(shuffledData, 0.2);
var trainData = trainTestData.TrainSet;
var testData = trainTestData.TestSet;

// Подготовка и нормализация данных
var pipeline = mlContext.Transforms
    .Concatenate("Features",
        nameof(CarInfo.Model),
        nameof(CarInfo.KilometersDriven),
        nameof(CarInfo.Year),
        // nameof(CarInfo.Owner),
        nameof(CarInfo.FuelType),
        nameof(CarInfo.Transmission),
        // nameof(CarInfo.Insurance),
        nameof(CarInfo.CarCondition)
    )
    .Append(mlContext.Transforms.NormalizeMinMax("NormalizedFeatures", "Features"));

// Выбираем алгоритм для тренировки
var trainer = mlContext.Regression.Trainers.FastTree(nameof(CarInfo.SellingPrice), "NormalizedFeatures");
var trainedPipeline = pipeline.Append(trainer);

// Обучение
ITransformer model = trainedPipeline.Fit(trainData);

// Тестирование
var crossValidationResults = mlContext.Regression.CrossValidate(trainData, trainedPipeline, 5,
    nameof(CarInfo.SellingPrice));
Console.WriteLine($"Cross-validated accuracy: {crossValidationResults.Average(cv => cv.Metrics.RSquared):P2}");

var predictions = model.Transform(testData);
var metrics = mlContext.Regression.Evaluate(predictions, nameof(CarInfo.SellingPrice));
Console.WriteLine($"Model accuracy: {metrics.RSquared:P2}");

// Предсказание
var predEngine = mlContext.Model.CreatePredictionEngine<CarInfo, CarSellingPricePrediction>(model);
var sample = MapToCarInfo("MarutiAltoK10,???,33639,2015,First Owner,Petrol,MANUAL,17-10-2021, 4.1");
var prediction = predEngine.Predict(sample);
Console.WriteLine($"{prediction.Score}");

CarInfo MapToCarInfo(string carInfoString)
{
    var carInfo = carInfoString.ToLowerInvariant().Split(',', StringSplitOptions.TrimEntries);
    var carModel = carModels.TryAdd(carInfo[0], carModels.Count + 1) ? carModels.Count : carModels[carInfo[0]];
    float.TryParse(carInfo[1], out var sellingPrice);
    float.TryParse(carInfo[2], out var kilometersDriven);
    float.TryParse(carInfo[3], out var year);
    year = DateTime.Now.Year - year;
    var owner = carInfo[4] switch
    {
        "first owner" => 1,
        "second owner" => 2,
        "third owner" => 3,
        _ => 0
    };
    var fuelType = fuelTypes.TryAdd(carInfo[5], fuelTypes.Count + 1) ? fuelTypes.Count : fuelTypes[carInfo[5]];
    var transmission = transmissions.TryAdd(carInfo[6], transmissions.Count + 1)
        ? transmissions.Count
        : transmissions[carInfo[6]];
    var date = DateTime.TryParse(carInfo[7], out var dateTime)
        ? dateTime.ToOADate()
        : DateTime.TryParseExact(carInfo[7], "dd-MM-yyyy", CultureInfo.InvariantCulture, DateTimeStyles.None,
            out var dateTime2)
            ? dateTime2.ToOADate()
            : 0;
    var insurance = (float)date;
    float.TryParse(carInfo[8], out var carCondition);

    return new CarInfo()
    {
        Model = carModel,
        KilometersDriven = kilometersDriven,
        Year = year,
        Owner = owner,
        FuelType = fuelType,
        Transmission = transmission,
        Insurance = insurance,
        CarCondition = carCondition,
        SellingPrice = sellingPrice
    };
}

internal class CarInfo
{
    [LoadColumn(0)] public float Model { get; set; }
    [LoadColumn(1)] public float KilometersDriven { get; set; }
    [LoadColumn(2)] public float Year { get; set; }
    [LoadColumn(3)] public float Owner { get; set; }
    [LoadColumn(4)] public float FuelType { get; set; }
    [LoadColumn(5)] public float Transmission { get; set; }
    [LoadColumn(6)] public float CarCondition { get; set; }
    [LoadColumn(7)] public float Insurance { get; set; }
    [LoadColumn(8)] public float SellingPrice { get; set; }

    public override string ToString()
    {
        return
            $"{Model},{SellingPrice},{KilometersDriven},{Year},{Owner},{FuelType},{Transmission},{Insurance},{CarCondition}";
    }
}

internal class CarSellingPricePrediction
{
    public float Score { get; set; }
}