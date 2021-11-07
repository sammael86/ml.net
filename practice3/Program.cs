using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;
// using MathNet.Numerics;

var mlContext = new MLContext(20210930);

var path = "phone-calls.csv";

var dailyRecordInfos = File.ReadLines(path)
    .Skip(1)
    .Select(MapToDailyRecord)
    .ToArray();

var dataView = mlContext.Data.LoadFromEnumerable(dailyRecordInfos);

var seasonality = mlContext.AnomalyDetection.DetectSeasonality(dataView, nameof(DailyRecord.value));

// var f = Directory.GetCurrentDirectory();
//
// Control.NativeProviderPath = "mathnet";
// Control.UseNativeMKL();

var result = mlContext.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, "out", nameof(DailyRecord.value),
    new SrCnnEntireAnomalyDetectorOptions
    {
        Threshold = 0.05,
        BatchSize = -1,
        Period =  seasonality,
        Sensitivity = 70,
        DetectMode = SrCnnDetectMode.AnomalyAndMargin,
        DeseasonalityMode = SrCnnDeseasonalityMode.Mean
    });

var deb = result.Preview();
;
DailyRecord MapToDailyRecord(string dailyInfoString)
{
    // var dailyRecordInfo = dailyInfoString.Split(',', StringSplitOptions.TrimEntries);
    // DateTime.TryParseExact(dailyRecordInfo[1], "yyyy/M/d", CultureInfo.InvariantCulture, DateTimeStyles.None,
    //     out var dateTime);
    // var date = dateTime.ToOADate();
    return new DailyRecord()
    {
        // timestamp = (float)date,
        value = dailyInfoString[1],
    };
}

internal class DailyRecord
{
    // [LoadColumn(0)] public float timestamp { get; set; }
    [LoadColumn(1)] public double value { get; set; }
}