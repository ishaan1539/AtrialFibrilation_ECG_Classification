%% 1. Low-pass filter the Challenge 2017 recordings at 40 Hz
% Run the sections in order. Requires Signal Processing Toolbox and
% Statistics and Machine Learning Toolbox.
% Run Section can execute a temporary copy outside the project folder.
projectCandidates = [string(fileparts(mfilename('fullpath'))), ...
    string(fileparts(which('ecgProjectAnalysis'))), string(pwd), ...
    fullfile(string(getenv('HOME')), "Desktop", "Sem 7", "BSE665", "Project_ECG")];
projectIndex = find(isfile(fullfile(projectCandidates, "training2017", "REFERENCE.csv")), 1);
assert(~isempty(projectIndex), ...
    "Could not locate training2017/REFERENCE.csv. Set MATLAB's Current Folder to Project_ECG.");
projectFolder = projectCandidates(projectIndex);
trainingFolder = fullfile(projectFolder, "training2017");
butFolder = fullfile(projectFolder, "butqdb");
recordedFile = fullfile(projectFolder, "recordings", "ecg_20261123_151617_625.mat");
recordedFs = 200;                % Assumed acquisition rate from the project comparison
adcReferenceV = 5;              % Assumed Arduino ADC reference
adcFullScaleCount = 1023;
modelFs = 300;
cutoffHz = 40;

classes = ["N", "A", "O", "~"];
featureNames = ["MedianBPM", "RRRelativeMAD", "RMSDeltaRR", ...
    "FractionDeltaRR50ms", "RRPercentileRange", "BeatSimilarity", ...
    "PWindowConsistency", "BaselineFraction", "HighFrequencyFraction", ...
    "FlatDifferenceFraction", "QRSContrast"];

reference = readtable(fullfile(trainingFolder, "REFERENCE.csv"), ...
    ReadVariableNames=false, TextType="string");
ids = strtrim(reference.Var1);
labels = categorical(strtrim(reference.Var2), classes);
assert(numel(unique(ids)) == numel(ids) && all(~isundefined(labels)), ...
    "Expected unique recording IDs and known rhythm labels in REFERENCE.csv.");

filteredTraining = cell(numel(ids), 1);
features = zeros(numel(ids), numel(featureNames));
[lowB, lowA] = butter(2, cutoffHz/(modelFs/2), "low");
[qrsB, qrsA] = butter(2, [5 20]/(modelFs/2), "bandpass");

for i = 1:numel(ids)
    recording = load(fullfile(trainingFolder, ids(i) + ".mat"), "val");
    header = splitlines(string(fileread(fullfile(trainingFolder, ids(i) + ".hea"))));
    fields = split(strtrim(header(1)));
    raw = double(recording.val(:));
    assert(str2double(fields(2)) == 1 && str2double(fields(3)) == modelFs && ...
        str2double(fields(4)) == numel(raw), "Unexpected record header for %s.", ids(i));

    filtered = filtfilt(lowB, lowA, raw);
    filteredTraining{i} = single(filtered);
    features(i,:) = forestFeatures(raw, filtered, modelFs, qrsB, qrsA);

    if mod(i, 1000) == 0 || i == numel(ids)
        fprintf("Filtered and extracted features from %d of %d recordings.\n", i, numel(ids));
    end
end

%% 2. Train ForestFull with a stratified 80:20 split and report its results
rng(42, "twister");
partition = cvpartition(labels, Holdout=0.20);
trainRows = training(partition);
testRows = test(partition);
order = categorical(classes(:), classes);
assert(all(countcats(labels(trainRows)) > 0) && all(countcats(labels(testRows)) > 0), ...
    "Both partitions must contain every rhythm class.");

forestFull = TreeBagger(150, features(trainRows,:), labels(trainRows), ...
    Method="classification", MinLeafSize=5, Prior="uniform", ...
    PredictorNames=cellstr(featureNames));
[testLabels, testScores] = predict(forestFull, features(testRows,:));
predicted = categorical(string(testLabels), classes);
actual = labels(testRows);
cm = confusionmat(actual, predicted, Order=order);
tp = diag(cm);
precision = tp ./ max(sum(cm, 1)', 1);
recall = tp ./ max(sum(cm, 2), 1);
f1 = 2 * precision .* recall ./ max(precision + recall, eps);

splitCounts = table(order, countcats(labels(trainRows)), countcats(actual), ...
    VariableNames=["Label", "TrainCount", "TestCount"]);
confusion = array2table(cm, RowNames=cellstr(classes(:)), ...
    VariableNames=["PredN", "PredA", "PredO", "PredNoise"]);
perClass = table(order, precision, recall, f1, VariableNames=["Label", "Precision", "Recall", "F1"]);
metrics = table(sum(tp)/sum(cm, "all"), precision(2), recall(2), f1(2), ...
    mean(f1(1:3)), mean(f1), VariableNames=["Accuracy", "AFPrecision", "AFRecall", ...
    "AFF1", "ChallengeF1", "MacroF1"]);
testPredictions = table(ids(testRows), actual, predicted, ...
    VariableNames=["Recording", "Actual", "Predicted"]);

fprintf("\nForestFull with the 40 Hz low-pass and REFERENCE.csv\n");
disp(splitCounts)
disp(confusion)
disp(perClass)
disp(metrics)

%% 3. Analyze BUT QDB Class 1 and predict with ForestFull
[class1Raw, class1Fs, class1Start] = readBUTExample(butFolder, 1);
class1 = analyzeTrace(class1Raw, class1Fs, cutoffHz, false, ...
    sprintf("BUT QDB Class 1 at %.3f s", class1Start), "\muV");
class1Predictions = comparePredictions(forestFull, class1Raw, class1Fs, ...
    modelFs, lowB, lowA, qrsB, qrsA);

%% 4. Analyze BUT QDB Class 2 and predict with ForestFull
[class2Raw, class2Fs, class2Start] = readBUTExample(butFolder, 2);
class2 = analyzeTrace(class2Raw, class2Fs, cutoffHz, false, ...
    sprintf("BUT QDB Class 2 at %.3f s", class2Start), "\muV");
class2Predictions = comparePredictions(forestFull, class2Raw, class2Fs, ...
    modelFs, lowB, lowA, qrsB, qrsA);

%% 5. Analyze the recorded ECG and estimate heart rate
recorded = load(recordedFile, "adcCounts");
recordedVolts = double(recorded.adcCounts(:)) * (adcReferenceV / adcFullScaleCount);
ownECG = analyzeTrace(recordedVolts, recordedFs, cutoffHz, true, ...
    sprintf("Recorded ECG at assumed %g Hz (ADC input)", recordedFs), "V", [45 65]);


% Local functions

function [raw, fs, startSeconds] = readBUTExample(folder, quality)
    excerpt = load(fullfile(folder, sprintf("100001_class%d.mat", quality)));
    fs = excerpt.sampleRateHz;
    assert(excerpt.qualityClass == quality && fs == 1000 && ...
        numel(excerpt.adcCounts) == 30*fs, "Unexpected Class %d ECG excerpt.", quality);
    startSeconds = (excerpt.startSample - 1) / fs;
    raw = (double(excerpt.adcCounts(:)) - excerpt.baselineCounts) / ...
        excerpt.gainCountsPerMicrovolt;
end

function result = analyzeTrace(raw, fs, cutoff, invertFiltered, name, unit, bandstopHz)
    if nargin < 7
        bandstopHz = [];
    end
    raw = raw(:);
    [b, a] = butter(2, cutoff/(fs/2), "low");
    filtered = filtfilt(b, a, raw);
    filteredName = "40 Hz low-pass";
    if invertFiltered
        filtered = 2 * median(filtered) - filtered;
        filteredName = filteredName + ", inverted";
    end

    signals = [raw, filtered];
    stage = ["Raw"; filteredName];
    if ~isempty(bandstopHz)
        [stopB, stopA] = butter(2, bandstopHz/(fs/2), "stop");
        signals(:,3) = filtfilt(stopB, stopA, filtered);
        stage(3,1) = sprintf("%g Hz LP + %g–%g Hz stop", ...
            cutoff, bandstopHz(1), bandstopHz(2));
    end

    [qrsB, qrsA] = butter(2, [5 20]/(fs/2), "bandpass");
    peakLocations = cell(numel(stage), 1);
    bpm = NaN(numel(stage), 1);
    for i = 1:numel(stage)
        peakLocations{i} = qrsPeaks(signals(:,i), fs, qrsB, qrsA);
        bpm(i) = estimatedBPM(peakLocations{i}, fs);
    end
    detectedBeats = cellfun(@numel, peakLocations);
    heartRate = table(stage, detectedBeats, bpm, ...
        VariableNames=["Signal", "DetectedQRSCandidates", "EstimatedBPM"]);
    fprintf("\n%s\n", name);
    disp(heartRate)

    t = (0:numel(raw)-1)' / fs;
    windowLength = min(numel(raw), round(2*fs));
    window = hamming(windowLength, "periodic");
    % Remove DC before Welch. The default PSD for real data is one-sided.
    [psd, frequency] = pwelch(signals - mean(signals, 1), window, ...
        floor(windowLength/2), 2^nextpow2(windowLength), fs);

    figure("Name", name);
    nTraces = size(signals, 2);
    tiledlayout(2, nTraces, TileSpacing="compact");
    waveformAxes = gobjects(nTraces, 1);
    spectrumAxes = gobjects(nTraces, 1);
    for i = 1:nTraces
        waveformAxes(i) = nexttile(i);
        plot(t, signals(:,i));
        hold on
        peaks = peakLocations{i};
        plot(t(peaks), signals(peaks,i), "o", MarkerSize=4);
        title(sprintf("%s | estimated %.1f bpm", stage(i), bpm(i)));
        xlabel("Time within recording segment (s)");
        ylabel("Voltage (" + unit + ")");
        grid on

        spectrumAxes(i) = nexttile(i + nTraces);
        plot(frequency, 10*log10(max(psd(:,i), realmin)));
        xlim([0 min(150, fs/2)]);
        title(stage(i) + " | Welch PSD");
        xlabel("Frequency (Hz)");
        ylabel("PSD (dB re " + unit + "^2/Hz)");
        grid on
    end
    linkaxes(waveformAxes, "xy");
    linkaxes(spectrumAxes, "xy");
    sgtitle(name);
    result = struct("TimeSeconds", t, "RawVoltage", raw, "FilteredVoltage", filtered, ...
        "VoltageUnit", unit, "SampleRateHz", fs, "FrequencyHz", frequency, ...
        "RawPSD", psd(:,1), "FilteredPSD", psd(:,2), "HeartRate", heartRate, ...
        "RawQRSCandidates", peakLocations{1}, "FilteredQRSCandidates", peakLocations{2});
    if ~isempty(bandstopHz)
        result.BandstopHz = bandstopHz;
        result.LowpassBandstopVoltage = signals(:,3);
        result.LowpassBandstopPSD = psd(:,3);
        result.LowpassBandstopQRSCandidates = peakLocations{3};
    end
end

function predictions = comparePredictions(model, raw, fs, modelFs, lowB, lowA, qrsB, qrsA)
    % Compute inference features at the same sampling rate as training.
    [p, q] = rat(modelFs / fs);
    atModelRate = resample(raw(:), p, q);
    modelFiltered = filtfilt(lowB, lowA, atModelRate);
    % The bypass is a sensitivity check outside the training preprocessing.
    x = [forestFeatures(atModelRate, atModelRate, modelFs, qrsB, qrsA);
         forestFeatures(atModelRate, modelFiltered, modelFs, qrsB, qrsA)];
    [label, score] = predict(model, x);
    stages = ["40 Hz bypass (experimental)"; "40 Hz low-pass (training pipeline)"];
    names = string(model.ClassNames);
    predictions = table(stages, string(label(:)), score(:,names=="N"), ...
        score(:,names=="A"), score(:,names=="O"), score(:,names=="~"), ...
        VariableNames=["Input", "PredictedRhythm", "ScoreN", "ScoreA", "ScoreO", "ScoreNoise"]);
    disp(predictions)
end

function bpm = estimatedBPM(peaks, fs)
    bpm = NaN;
    if numel(peaks) >= 3
        bpm = 60 / median(diff(peaks)/fs);
    end
end

function [peaks, qrs] = qrsPeaks(signal, fs, b, a)
    % ForestFull detector: locate QRS energy and refine with absolute amplitude.
    qrs = filtfilt(b, a, signal);
    energy = movmean([0; diff(qrs)].^2, round(0.10*fs));
    baseEnergy = median(energy);
    threshold = baseEnergy + 0.12*(prctile(energy, 98) - baseEnergy);
    [~, candidates] = findpeaks(energy, MinPeakHeight=max(threshold, eps), ...
        MinPeakDistance=round(0.24*fs));
    peaks = zeros(size(candidates));
    for i = 1:numel(candidates)
        interval = max(1, candidates(i)-round(0.08*fs)) : ...
            min(numel(signal), candidates(i)+round(0.08*fs));
        [~, offset] = max(abs(qrs(interval)));
        peaks(i) = interval(offset);
    end
    peaks = unique(peaks);
end

function f = forestFeatures(raw, filtered, fs, b, a)
    % The eleven predictors in the project's ForestFull model.
    [peaks, qrs] = qrsPeaks(filtered, fs, b, a);
    rr = diff(peaks)/fs;
    f = zeros(1, 11);
    if ~isempty(rr)
        typical = median(rr);
        f(1:2) = [60/typical, median(abs(rr-typical))/typical];
        f(5) = diff(prctile(rr, [10 90]));
    end
    if numel(rr) >= 2
        delta = diff(rr);
        f(3:4) = [sqrt(mean(delta.^2)), mean(abs(delta) > 0.05)];
    end
    usable = peaks(peaks > round(0.24*fs) & peaks+round(0.10*fs) <= numel(filtered));
    f(6) = templateSimilarity(filtered, usable, -round(0.08*fs):round(0.10*fs));
    f(7) = templateSimilarity(filtered, usable, -round(0.24*fs):-round(0.10*fs));
    baseline = movmean(raw, round(fs));
    scale = max(std(raw), eps);
    f(8:11) = [std(baseline)/scale, std(raw-filtered-baseline)/scale, ...
        mean(diff(raw)==0), prctile(abs(qrs),99)/max(median(abs(qrs)),eps)];
end

function similarity = templateSimilarity(x, peaks, offsets)
    similarity = 0;
    if numel(peaks) < 3
        return
    end
    beats = x(peaks(:) + offsets);
    beats = beats - mean(beats, 2);
    beats = beats ./ max(vecnorm(beats, 2, 2), eps);
    template = median(beats, 1);
    similarity = median(beats * template' / max(norm(template), eps));
end
