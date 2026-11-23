%% Load a dataset

load("training2017/A00001.mat");
fs = 300;
t = (0:numel(val)-1)/fs;
plot(t, val);
xlim([10 15]);
xlabel('Time (s)');
ylabel('Signal');
grid on;

%% Which recordings are atrial fibrilations
reference = readcell("training2017/REFERENCE.csv", "Delimiter", ",");
recordings = string(reference(:,1));
labels = upper(strtrim(string(reference(:,2))));

atrialFibrillation = recordings(labels == "A");
disp("Recordings labelled as atrial fibrillation:");
disp(atrialFibrillation);


%% ECG Recording

acquisition = load("recordings/ecg_20261123_151617_625.mat", ...
    "time", "adcCounts", "nominalSampleRateHz");
referenceECG = load("training2017/A00001.mat", "val");
referenceFs = 300;
referenceSamples = (10 * referenceFs + 1):(20 * referenceFs);
referenceTime = (0:numel(referenceSamples)-1)' / referenceFs;
arduinoModelTime = double(acquisition.time(:));
arduinoModelTime = arduinoModelTime - arduinoModelTime(1);

figure('Name', 'ECG recordings as acquired');
tiledlayout(2, 1);

referenceAxes = nexttile;
plot(referenceTime, referenceECG.val(referenceSamples));
title('PhysioNet A00001 (10–20 s of source)');
xlabel('Recording time within window (s)');
ylabel('PhysioNet ADC counts');
grid on;

recordedAxes = nexttile;
plot(arduinoModelTime, acquisition.adcCounts(:));
title(sprintf('Arduino ECG (nominal %g samples/s)', acquisition.nominalSampleRateHz));
xlabel('Simulink model time (s)');
ylabel('Arduino ADC counts');
grid on;
linkaxes([referenceAxes recordedAxes], 'x');



%% Processing

recording = load("recordings/ecg_20261123_151617_625.mat", ...
    "time", "adcCounts", "nominalSampleRateHz");
fs = recording.nominalSampleRateHz;
volts = double(recording.adcCounts(:)) * (5 / 1023);  % Assumes a 5 V ADC reference.

windowSamples = min(numel(volts), round(2 * fs));
window = hamming(windowSamples, 'periodic');
overlapSamples = floor(windowSamples / 2);
fftSamples = 2^nextpow2(windowSamples);
[PSD_welch, PSD_f] = pwelch(volts - mean(volts), window, ...
    overlapSamples, fftSamples, fs);

figure('Name', 'Arduino ECG and Welch PSD');
tiledlayout(2, 1);

nexttile;
plot(recording.time, volts);
title('Arduino ECG');
xlabel('Simulink model time (s)');
ylabel('Voltage (V)');
grid on;

nexttile;
plot(PSD_f, 10 * log10(max(PSD_welch, realmin)));
title('Welch PSD after removing the DC offset');
xlabel('Frequency (Hz, nominal)');
ylabel('PSD (dB re 1 V^2/Hz)');
grid on;
