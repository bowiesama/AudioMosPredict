import CreateML
import Foundation

let audioData = try MLDataTable(contentsOf: URL(fileURLWithPath: "/private/tmp/allInOne.csv"))
let param = CreateML.MLRandomForestRegressor.ModelParameters(maxDepth: 10, maxIterations: 2000)
let (trainingCSVData, testCSVData) = audioData.randomSplit(by: 0.7, seed: 0)

//let pricer = try MLRegressor(trainingData: audioData, targetColumn: "netMosAvg")
let pricer = try MLRandomForestRegressor(trainingData: audioData, targetColumn: "MOSLQ", parameters:param)
let csvMetadata = MLModelMetadata(author: "bowchen", shortDescription: "A Model Predict Audio Mos", version: "2.0")
try pricer.write(to: URL(fileURLWithPath: "/Users/wme/git/AudioMosPredict/Audio_Mos_Predict/Audio_mos1.mlmodel"), metadata: csvMetadata)
print(pricer)
//let audioTestData = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/bowchen/git/AudioMosPredict/Audio_Mos_Predict/Testdata.csv"))
//let result = try pricer.predictions(from: audioTestData)
//print(result)
