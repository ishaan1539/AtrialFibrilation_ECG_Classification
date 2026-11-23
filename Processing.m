%% Load a dataset

load("training2017/A00001.mat");
fs = 300;
t = (0:numel(val)-1)/fs;
plot(t, val);

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

%% AD8232 Bluetooth ECG

bluetoothECG = readmatrix("ECG Bluetooth Board A.csv");
bluetoothFs = 250;  % Sampling rate reported for the Bluetooth board.
bluetoothTime = (0:numel(bluetoothECG)-1)' / bluetoothFs;

figure('Name', 'AD8232 Bluetooth ECG');
plot(bluetoothTime, bluetoothECG);
xlim([10 15]);
xlabel('Time (s)');
ylabel('ADC counts');
grid on;


%% Data Processing 

load("recordings/ecg_20261123_151617_625.mat");
adcCounts = adcCounts - mean(adcCounts);
fs = 1000;
%plot(time, adcCounts);
%xlabel('Time (s)');
%ylabel('Signal');
grid on;
subplot(2,1,1);
y_n = filter(bandpass_filter1,1,adcCounts);
[PSD_welch, PSD_f] = pwelch(y_n,[],[],1000,fs);
plot(PSD_f,2*PSD_welch);

subplot(2,1,2);
plot(time, y_n);



s = load("recordings/ecg_20261123_151617_625_filter_comparison.mat");
figure
plot(s.time, s.bandpass0p5to40)
xlim([2, 4]);
xlabel("Simulink model time (s)")
ylabel("Filtered ADC counts")
grid on


%% Low pass 150 HZ

s = load("recordings/ecg_20261123_151617_625.mat");
fs = s.nominalSampleRateHz;
time = s.time(:);
adcCounts = double(s.adcCounts(:));

% Zero-phase Butterworth low-pass for offline analysis.
[b, a] = butter(2, 150/(fs/2), "low");
lowpass150 = filtfilt(b, a, adcCounts);

save("recordings/ecg_20261123_151617_625_lowpass150.mat", ...
    "time", "fs", "lowpass150");

figure
plot(time, lowpass150)
xlim([2 4])
xlabel("Simulink model time (s)")
ylabel("Filtered ADC counts")
title("ECG low-pass filtered at 150 Hz")
grid on

%% Low Pass 40 Hz, 200 Hz Fs

s = load("recordings/ecg_20261123_151617_625_lowpass40_fs200.mat");
plot(s.timeSeconds, s.lowpass40)
xlim([10 20])
xlabel("Time (s), assuming 200 Hz")
ylabel("Filtered ADC counts")
grid on


%% Comparisons

addpath("analysis")
[~, ref] = analyzeBUTQDB;  % Loads the Class 1 and Class 2 examples

% Load your original recording and filter the full trace before cropping.
s = load("recordings/ecg_20261123_151617_625.mat");
fs = 200;  % Assumed sampling rate for your recording
myRaw = double(s.adcCounts(:));
[b, a] = butter(2, 40/(fs/2), "low");
myFiltered = filtfilt(b, a, myRaw);

% Show 10–20 s of your recording, with the display axis starting at zero.
idx = 10*fs + (1:10*fs);
myTime = (0:numel(idx)-1)' / fs;
myInverted = median(myFiltered(idx)) - myFiltered(idx);

rowNames = ["My recording", "BUT QDB Class 1", "BUT QDB Class 2"];
times = {myTime, ref(1).TimeSeconds, ref(2).TimeSeconds};
raw = {myRaw(idx), ref(1).RawMicrovolts, ref(2).RawMicrovolts};
filtered = {myInverted, ref(1).FilteredMicrovolts, ref(2).FilteredMicrovolts};
rawUnits = ["ADC counts", "\muV", "\muV"];
filteredUnits = ["Centered ADC counts", "\muV", "\muV"];
filterLabels = ["40 Hz low-pass, inverted", "40 Hz low-pass", "40 Hz low-pass"];

figure("Name", "ECG comparison");
tiledlayout(3, 2, "TileSpacing", "compact");
ax = gobjects(6, 1);

for k = 1:3
    ax(2*k-1) = nexttile;
    plot(times{k}, raw{k});
    title(rowNames(k) + " | Raw");
    ylabel(rawUnits(k));
    grid on;

    ax(2*k) = nexttile;
    plot(times{k}, filtered{k});
    title(rowNames(k) + " | " + filterLabels(k));
    ylabel(filteredUnits(k));
    grid on;
end

linkaxes(ax, "x");
xlim(ax(1), [0 10]);
xlabel(ax(5), "Seconds within window");
xlabel(ax(6), "Seconds within window");