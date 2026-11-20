function results = trainAFDetector(options)
% Train and evaluate AF classifiers on the revised Challenge 2017 labels.
% Run results = trainAFDetector from MATLAB. For a short integration check,
% use trainAFDetector(MaxPerClass=24, Trees=8, Epochs=1).
% Enable XGBoost with RunXGBoost=true and xgboost installed in MATLAB's
% configured Python environment (see pyenv).
arguments
    options.DataDir (1,1) string = fullfile(fileparts(mfilename('fullpath')), "training2017")
    options.MaxPerClass (1,1) double {mustBePositive} = Inf
    options.Trees (1,1) double {mustBeInteger,mustBePositive} = 150
    options.Epochs (1,1) double {mustBeInteger,mustBePositive} = 12
    options.RunCNN (1,1) logical = true
    options.RunTunedForest (1,1) logical = true
    options.RunResNet (1,1) logical = true
    options.RunXGBoost (1,1) logical = false
end

rng(42, "twister");
classes = ["N","A","O","~"];
reference = readtable(fullfile(options.DataDir,"REFERENCE.csv"), ...
    ReadVariableNames=false, TextType="string");
ids = reference.Var1;
labels = categorical(reference.Var2, classes);
assert(numel(unique(ids)) == numel(ids) && all(~isundefined(labels)), ...
    "Expected unique recording IDs and four supported labels.");

if isfinite(options.MaxPerClass)
    assert(options.MaxPerClass >= 12 && mod(options.MaxPerClass,1) == 0, ...
        "MaxPerClass must be an integer of at least 12.");
    selected = false(size(ids));
    for label = classes
        group = find(labels == label);
        selected(group(randperm(numel(group),min(numel(group),options.MaxPerClass)))) = true;
    end
    ids = ids(selected);
    labels = labels(selected);
end

partition = cvpartition(labels, Holdout=0.20);
trainRows = find(training(partition));
testRows = find(test(partition));
assert(all(countcats(labels(trainRows)) > 0) && all(countcats(labels(testRows)) > 0), ...
    "Both splits must contain all four labels.");

fs = 300;
waveformStride = 2;
maxSamples = ceil(61*fs/waveformStride);
[b,a] = butter(2,[0.5 40]/(fs/2),"bandpass");
[qb,qa] = butter(2,[5 20]/(fs/2),"bandpass");
featureNames = ["MedianBPM","RRRelativeMAD","RMSDeltaRR","FractionDeltaRR50ms", ...
    "RRPercentileRange","BeatSimilarity","PWindowConsistency","BaselineFraction", ...
    "HighFrequencyFraction","FlatDifferenceFraction","QRSContrast"];
baselineColumns = numel(featureNames);
if options.RunTunedForest
    featureNames = [featureNames,"RRShortLongFraction","RRLagOneCorrelation", ...
        "RRRobustSpread","RRDeltaMedian","RRRelativeRMSSD","RRAlternation"];
end
features = zeros(numel(ids),numel(featureNames));
if options.RunCNN
    cnnSamples = 8*fs/waveformStride;
    windowsPerRecording = 3;
    cnnWindows = cell(numel(ids),windowsPerRecording);
end
if options.RunResNet
    images = zeros(maxSamples,1,1,numel(ids),"single");
end
for j = 1:numel(ids)
    record = load(fullfile(options.DataDir,ids(j)+".mat"),"val");
    header = splitlines(string(fileread(fullfile(options.DataDir,ids(j)+".hea"))));
    metadata = split(strtrim(header(1)));
    assert(str2double(metadata(2)) == 1 && str2double(metadata(3)) == fs && ...
        str2double(metadata(4)) == numel(record.val), "Unexpected header for %s.",ids(j));
    raw = double(record.val(:));
    filtered = filtfilt(b,a,raw);
    [base,extra] = ecgFeatures(raw,filtered,fs,qb,qa);
    features(j,1:baselineColumns) = base;
    if options.RunTunedForest
        features(j,baselineColumns+1:end) = extra;
    end
    if options.RunResNet || options.RunCNN
        % Keep the entire clip. Center and scale before zero padding.
        % The 40 Hz low-pass also attenuates frequencies above the 75 Hz
        % Nyquist frequency of the downsampled waveform.
        waveform = resample(filtered,1,waveformStride);
        waveform = waveform - median(waveform);
        assert(numel(waveform) <= maxSamples,"Recording %s exceeds 61 seconds.",ids(j));
        waveform = single(waveform / max(prctile(abs(waveform),95),eps));
        if options.RunResNet
            images(1:numel(waveform),1,1,j) = waveform;
        end
        if options.RunCNN
            assert(numel(waveform) >= cnnSamples,"Recording %s is shorter than 8 seconds.",ids(j));
            starts = round(linspace(1,numel(waveform)-cnnSamples+1,windowsPerRecording));
            for windowIndex = 1:windowsPerRecording
                samples = starts(windowIndex)+(0:cnnSamples-1);
                cnnWindows{j,windowIndex} = reshape(waveform(samples),1,[]);
            end
        end
    end
end

results.Split = table(ids,labels,strings(size(ids)), ...
    VariableNames=["Recording","Label","Partition"]);
results.Split.Partition(trainRows) = "train";
results.Split.Partition(testRows) = "test";
results.Features = array2table(features,VariableNames=featureNames);
results.Filter = struct('SampleRate',fs,'B',b,'A',a,'QrsB',qb,'QrsA',qa);
results.ClassOrder = classes;
results.ResNetInput = struct('SampleRate',fs/waveformStride,'Samples',maxSamples);
results.Predictions = table(ids(testRows),labels(testRows), ...
    VariableNames=["Recording","Actual"]);
results.Metrics = table();
actual = labels(testRows);
order = categorical(classes,classes);

for modelName = ["ForestRR","ForestFull","BoostedTrees"]
    columns = 1:baselineColumns;
    if modelName == "ForestRR"
        columns = 1:5;
    end
    x = features(:,columns);
    if modelName == "BoostedTrees"
        mdl = fitcensemble(x(trainRows,:),labels(trainRows), ...
            Method="AdaBoostM2",NumLearningCycles=options.Trees, ...
            Learners=templateTree(MaxNumSplits=20,MinLeafSize=5),Prior="uniform");
    else
        mdl = TreeBagger(options.Trees,x(trainRows,:),labels(trainRows), ...
            Method="classification",MinLeafSize=5,Prior="uniform");
    end
    results.Models.(modelName) = mdl;
    results.ModelFeatures.(modelName) = featureNames(columns);
    results = addEvaluation(results,modelName,actual,predict(mdl,x(testRows,:)),order);
end

if options.RunTunedForest
    baselineRng = rng;
    rng(43,"twister");
    mdl = TreeBagger(options.Trees,features(trainRows,:),labels(trainRows), ...
        Method="classification",MinLeafSize=5,Prior="uniform",OOBPrediction="on");
    [~,trainingScores] = oobPredict(mdl);
    hasOOB = any(mdl.OOBIndices,2) & all(isfinite(trainingScores),2);
    classNames = string(mdl.ClassNames);
    afColumn = find(classNames == "A");
    thresholds = 0.20:0.025:0.80;
    bestF1 = -Inf;
    threshold = 0.5;
    for candidate = thresholds
        predictedAF = trainingScores(hasOOB,afColumn) >= candidate;
        actualAF = labels(trainRows(hasOOB)) == "A";
        f1 = 2*nnz(predictedAF & actualAF)/max(nnz(predictedAF)+nnz(actualAF),1);
        if f1 > bestF1
            bestF1 = f1;
            threshold = candidate;
        end
    end
    [~,scores] = predict(mdl,features(testRows,:));
    alternativeScores = scores;
    alternativeScores(:,afColumn) = -Inf;
    [~,prediction] = max(alternativeScores,[],2);
    prediction(scores(:,afColumn) >= threshold) = afColumn;
    results.Models.ForestTuned = mdl;
    results.ModelFeatures.ForestTuned = featureNames;
    results.ForestTuning = struct('AFThreshold',threshold,'OOB_AFF1',bestF1, ...
        'OOBRecordings',nnz(hasOOB));
    results = addEvaluation(results,"ForestTuned",actual,classNames(prediction),order);
    rng(baselineRng);
end

if options.RunXGBoost
    counts = countcats(labels(trainRows));
    [~,labelIndex] = ismember(string(labels(trainRows)),classes);
    weights = numel(trainRows) ./ (numel(classes)*counts(labelIndex));
    [mdl,prediction] = pyrun(strjoin([
        "import numpy as np"
        "from xgboost import XGBClassifier"
        "model = XGBClassifier(n_estimators=int(trees), max_depth=4, learning_rate=0.05, random_state=42)"
        "model.fit(np.asarray(x_train), np.asarray(y_train, dtype=np.int32).ravel(), sample_weight=np.asarray(w).ravel())"
        "prediction = model.predict(np.asarray(x_test)).tolist()"
        ],newline),["model","prediction"],x_train=features(trainRows,1:baselineColumns), ...
        y_train=int32(labelIndex-1),w=weights,x_test=features(testRows,1:baselineColumns),trees=options.Trees);
    predictedIndex = cellfun(@double,cell(prediction)) + 1;
    results.Models.XGBoost = mdl;
    results.ModelFeatures.XGBoost = featureNames(1:baselineColumns);
    results = addEvaluation(results,"XGBoost",actual,classes(predictedIndex),order);
end

if options.RunResNet
    net = makeResNet(maxSamples);
    % Balance the training loss while preserving evaluation prevalence.
    weights = numel(trainRows)./(numel(classes)*countcats(labels(trainRows)));
    loss = @(Y,T) crossentropy(Y,T,single(weights),WeightsFormat="C");
    config = trainingOptions("adam",MaxEpochs=options.Epochs,MiniBatchSize=64, ...
        Shuffle="every-epoch",OutputNetwork="last-iteration",Verbose=false);
    mdl = trainnet(images(:,:,:,trainRows),labels(trainRows),net,loss,config);
    scores = minibatchpredict(mdl,images(:,:,:,testRows));
    results.Models.ResNet = mdl;
    results = addEvaluation(results,"ResNet",actual,scores2label(scores,order),order);
end
if options.RunCNN
    rng(44,"twister");
    layers = [
        sequenceInputLayer(1,Normalization="none",MinLength=cnnSamples)
        convolution1dLayer(15,32,Stride=2,Padding="same")
        batchNormalizationLayer
        reluLayer
        maxPooling1dLayer(3,Stride=2)
        convolution1dLayer(11,48,Stride=2,Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution1dLayer(9,64,DilationFactor=4,Padding="same")
        batchNormalizationLayer
        reluLayer
        convolution1dLayer(7,64,DilationFactor=8,Padding="same")
        batchNormalizationLayer
        reluLayer
        globalAveragePooling1dLayer
        dropoutLayer(0.2)
        fullyConnectedLayer(numel(classes))
        softmaxLayer
    ];
    trainWindows = reshape(cnnWindows(trainRows,:),[],1);
    trainLabels = repmat(labels(trainRows),windowsPerRecording,1);
    % Use milder balancing than the ResNet to reduce pressure to overcall AF.
    weights = sqrt(numel(trainRows)./(numel(classes)*countcats(labels(trainRows))));
    loss = @(Y,T) crossentropy(Y,T,single(weights),WeightsFormat="C");
    config = trainingOptions("adam",MaxEpochs=options.Epochs,MiniBatchSize=64, ...
        InputDataFormats="CTB",Shuffle="every-epoch",Verbose=false);
    mdl = trainnet(trainWindows,trainLabels,layers,loss,config);
    testWindows = reshape(cnnWindows(testRows,:),[],1);
    scores = minibatchpredict(mdl,testWindows,InputDataFormats="CTB");
    recordingScores = squeeze(mean(reshape(scores,numel(testRows), ...
        windowsPerRecording,numel(classes)),2));
    results.Models.CNN1D = mdl;
    results.CNNInput = struct('SampleRate',fs/waveformStride, ...
        'WindowSamples',cnnSamples,'WindowsPerRecording',windowsPerRecording);
    results = addEvaluation(results,"CNN1D",actual,scores2label(recordingScores,order),order);
end
disp(results.Metrics);
end

function [f,extra] = ecgFeatures(raw,x,fs,b,a)
% Detect steep QRS energy, then refine timing on the local band-passed trace.
qrs = filtfilt(b,a,x);
energy = movmean([0;diff(qrs)].^2,round(0.10*fs));
threshold = median(energy) + 0.12*(prctile(energy,98)-median(energy));
[~,candidates] = findpeaks(energy,MinPeakHeight=max(threshold,eps), ...
    MinPeakDistance=round(0.24*fs));
peaks = zeros(size(candidates));
for i = 1:numel(candidates)
    window = max(1,candidates(i)-round(.08*fs)):min(numel(x),candidates(i)+round(.08*fs));
    [~,offset] = max(abs(qrs(window)));
    peaks(i) = window(offset);
end
peaks = unique(peaks);
rr = diff(peaks)/fs;
f = zeros(1,11);
extra = zeros(1,6);
if ~isempty(rr)
    f(1:2) = [60/median(rr),median(abs(rr-median(rr)))/median(rr)];
    f(5) = diff(prctile(rr,[10 90]));
end
if numel(rr) >= 2
    delta = diff(rr);
    f(3:4) = [sqrt(mean(delta.^2)),mean(abs(delta) > .05)];
    typical = median(rr);
    left = rr(1:end-1);
    right = rr(2:end);
    centeredLeft = left-mean(left);
    centeredRight = right-mean(right);
    extra(1:5) = [mean(left < 0.8*typical & right > 1.2*typical), ...
        dot(centeredLeft,centeredRight)/max(norm(centeredLeft)*norm(centeredRight),eps), ...
        f(5)/typical,median(abs(delta))/typical,f(3)/typical];
    if numel(delta) >= 2
        extra(6) = mean(delta(1:end-1).*delta(2:end) < 0);
    end
end
usable = peaks(peaks > round(.24*fs) & peaks+round(.10*fs) <= numel(x));
f(6) = templateSimilarity(x,usable,-round(.08*fs):round(.10*fs));
% Similarity of activity before each QRS is a proxy for repeatable atrial activity.
f(7) = templateSimilarity(x,usable,-round(.24*fs):-round(.10*fs));
baseline = movmean(raw,round(fs));
f(8:11) = [std(baseline)/max(std(raw),eps), ...
    std(raw-x-baseline)/max(std(raw),eps),mean(diff(raw)==0), ...
    prctile(abs(qrs),99)/max(median(abs(qrs)),eps)];
end

function similarity = templateSimilarity(x,peaks,offsets)
similarity = 0;
if numel(peaks) < 3
    return
end
beats = x(peaks(:)+offsets);
beats = beats - mean(beats,2);
beats = beats ./ max(vecnorm(beats,2,2),eps);
template = median(beats,1);
similarity = median(beats*template'/max(norm(template),eps));
end

function net = makeResNet(maxSamples)
layers = [
    imageInputLayer([maxSamples 1 1],Normalization="none",Name="input")
    convolution2dLayer([15 1],24,Stride=[4 1],Padding="same",Name="stem")
    batchNormalizationLayer(Name="stemBN")
    reluLayer(Name="stemRelu")
];
net = addLayers(dlnetwork,layers);
previous = "stemRelu";
for block = 1:3
    name = "block"+block;
    residual = [
        convolution2dLayer([9 1],24,Padding="same",DilationFactor=[2^(block-1) 1],Name=name+"conv1")
        batchNormalizationLayer(Name=name+"bn1")
        reluLayer(Name=name+"relu1")
        convolution2dLayer([7 1],24,Padding="same",Name=name+"conv2")
        batchNormalizationLayer(Name=name+"bn2")
        additionLayer(2,Name=name+"add")
        reluLayer(Name=name+"out")
    ];
    net = addLayers(net,residual);
    net = connectLayers(net,previous,name+"conv1");
    net = connectLayers(net,previous,name+"add/in2");
    previous = name+"out";
end
net = addLayers(net,[
    globalAveragePooling2dLayer(Name="pool")
    fullyConnectedLayer(4,Name="logits")
    softmaxLayer(Name="probabilities")
]);
net = connectLayers(net,previous,"pool");
net = initialize(net);
end

function results = addEvaluation(results,name,actual,prediction,order)
prediction = categorical(string(prediction(:)),string(order));
cm = confusionmat(actual,prediction,Order=order);
tp = diag(cm);
precision = tp./max(sum(cm,1)',1);
recall = tp./max(sum(cm,2),1);
f1 = 2*precision.*recall./max(precision+recall,eps);
results.Confusion.(name) = array2table(cm, ...
    VariableNames=["PredN","PredA","PredO","PredNoise"],RowNames=cellstr(string(order)));
results.PerClass.(name) = table(order(:),precision,recall,f1, ...
    VariableNames=["Label","Precision","Recall","F1"]);
results.Metrics = [results.Metrics;table(name,precision(2),recall(2),f1(2), ...
    mean(f1(1:3)),mean(f1),VariableNames=["Model","AFPrecision","AFRecall","AFF1","ChallengeF1","MacroF1"])];
results.Predictions.(name) = prediction;
end
