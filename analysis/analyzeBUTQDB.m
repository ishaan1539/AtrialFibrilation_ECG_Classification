function [summary, examples] = analyzeBUTQDB(dataFolder, windowSeconds)
%ANALYZEBUTQDB Compare consensus Class 1 and Class 2 ECG in record 100001.
%   [summary, examples] = analyzeBUTQDB() reads the three files in Downloads.
%   analyzeBUTQDB(dataFolder, windowSeconds) selects another folder or a
%   different preview duration. Requires Signal Processing Toolbox.

arguments
    dataFolder (1,1) string = string(fullfile(getenv("HOME"), "Downloads"))
    windowSeconds (1,1) double {mustBeFinite, mustBeGreaterThanOrEqual(windowSeconds, 1)} = 10
end

headerPath = fullfile(dataFolder, "100001_ECG.hea");
waveformPath = fullfile(dataFolder, "100001_ECG.dat");
annotationPath = fullfile(dataFolder, "100001_ANN.csv");
files = [headerPath waveformPath annotationPath];
assert(all(isfile(files)), "Expected the header, ECG data and annotation CSV for 100001 in %s.", dataFolder);

header = splitlines(string(fileread(headerPath)));
recordFields = split(strtrim(header(1)));
signalFields = split(strtrim(header(2)));
fs = str2double(recordFields(3));
sampleCount = str2double(recordFields(4));
assert(str2double(recordFields(2)) == 1 && signalFields(2) == "16" && ...
    isfinite(fs) && fs > 80 && isfinite(sampleCount) && sampleCount > 0, ...
    "Expected a single-channel WFDB format 16 ECG sampled above 80 Hz.");
calibration = regexp(char(signalFields(3)), '^([\d.]+)\((-?\d+)\)/uV$', 'tokens', 'once');
assert(~isempty(calibration), "Expected WFDB gain in counts per microvolt and a baseline.");
gain = str2double(calibration{1});
baseline = str2double(calibration{2});
assert(gain > 0, "ECG calibration gain must be positive.");

annotations = readmatrix(annotationPath, "NumHeaderLines", 0);
assert(size(annotations, 2) == 12, "Expected 12 annotation columns.");
consensus = annotations(:, 10:12);
% Trailing rows can contain individual annotations but no consensus values.
consensus = consensus(~all(isnan(consensus), 2), :);
assert(all(isfinite(consensus), "all") && all(consensus == fix(consensus), "all") && ...
    all(consensus(:,1) >= 1 & consensus(:,2) >= consensus(:,1) & ...
        consensus(:,2) <= sampleCount & ismember(consensus(:,3), 0:3)), ...
    "Invalid consensus interval in the annotation CSV.");

fileInfo = dir(waveformPath);
assert(fileInfo.bytes == 2 * sampleCount, "Waveform size disagrees with the header.");
[fileID, message] = fopen(waveformPath, "r", "ieee-le");
assert(fileID >= 0, "Cannot open waveform file: %s", message);
closeWaveform = onCleanup(@() fclose(fileID)); %#ok<NASGU>

classes = [1; 2];
intervalCount = zeros(2, 1);
labeledMinutes = zeros(2, 1);
exampleStartSample = zeros(2, 1);
exampleEndSample = zeros(2, 1);
previewSamples = round(windowSeconds * fs);
paddingSamples = round(fs);
[b, a] = butter(2, 40 / (fs / 2), "low");

figure("Name", "BUT QDB 100001: Class 1 and Class 2");
tiledlayout(2, 2, "TileSpacing", "compact");
waveformAxes = gobjects(2, 1);
psdAxes = gobjects(2, 1);

for i = 1:numel(classes)
    qualityClass = classes(i);
    intervals = consensus(consensus(:,3) == qualityClass, 1:2);
    lengths = intervals(:,2) - intervals(:,1) + 1;
    intervalCount(i) = height(intervals);
    labeledMinutes(i) = sum(lengths) / fs / 60;
    selected = find(lengths >= previewSamples + 2 * paddingSamples, 1);
    assert(~isempty(selected), "No Class %d interval can fit a %.2f s preview with padding.", ...
        qualityClass, windowSeconds);

    firstReadSample = intervals(selected, 1);
    readCount = previewSamples + 2 * paddingSamples;
    assert(fseek(fileID, 2 * (firstReadSample - 1), "bof") == 0, "Could not seek to ECG interval.");
    counts = fread(fileID, readCount, "int16=>double");
    assert(numel(counts) == readCount, "Short waveform read.");
    paddedMicrovolts = (counts - baseline) / gain;
    filteredPadded = filtfilt(b, a, paddedMicrovolts);

    crop = paddingSamples + (1:previewSamples);
    rawMicrovolts = paddedMicrovolts(crop);
    filteredMicrovolts = filteredPadded(crop);
    timeSeconds = (0:previewSamples-1)' / fs;
    exampleStartSample(i) = firstReadSample + paddingSamples;
    exampleEndSample(i) = exampleStartSample(i) + previewSamples - 1;
    exampleStartSeconds = (exampleStartSample(i) - 1) / fs;

    psdWindow = min(previewSamples, round(2 * fs));
    taper = hann(psdWindow, "periodic");
    overlap = floor(psdWindow / 2);
    nfft = max(2, 2^nextpow2(psdWindow));
    [rawPSD, frequencyHz] = pwelch(rawMicrovolts - mean(rawMicrovolts), taper, overlap, nfft, fs);
    filteredPSD = pwelch(filteredMicrovolts - mean(filteredMicrovolts), taper, overlap, nfft, fs);

    examples(i,1) = struct("QualityClass", qualityClass, "SampleRateHz", fs, ...
        "StartSample", exampleStartSample(i), "EndSample", exampleEndSample(i), ...
        "TimeSeconds", timeSeconds, "RawMicrovolts", rawMicrovolts, ...
        "FilteredMicrovolts", filteredMicrovolts, "FrequencyHz", frequencyHz, ...
        "RawPSD", rawPSD, "FilteredPSD", filteredPSD); %#ok<AGROW>

    waveformAxes(i) = nexttile(i);
    % Subtract the same display offset from both curves.
    offset = median(rawMicrovolts);
    plot(timeSeconds, [rawMicrovolts filteredMicrovolts] - offset);
    title(sprintf("Class %d | record time %.3f s", qualityClass, exampleStartSeconds));
    xlabel("Seconds within interval");
    ylabel("Centered ECG (\muV)");
    legend("Raw", "40 Hz low-pass");
    grid on;

    psdAxes(i) = nexttile(i + 2);
    plot(frequencyHz, 10 * log10(max([rawPSD filteredPSD], realmin)));
    xlim([0 min(150, fs / 2)]);
    title(sprintf("Class %d | Welch PSD", qualityClass));
    xlabel("Frequency (Hz)");
    ylabel("PSD (dB re \muV^2/Hz)");
    grid on;
end
linkaxes(waveformAxes, "xy");
linkaxes(psdAxes, "xy");

summary = table(classes, intervalCount, labeledMinutes, exampleStartSample, exampleEndSample, ...
    VariableNames=["QualityClass", "IntervalCount", "LabeledMinutes", ...
        "ExampleStartSample", "ExampleEndSample"]);
disp(summary);
end
