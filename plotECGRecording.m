function filename = plotECGRecording(acquisition,options)
% Save the raw A0 samples as a MAT-file and optionally plot them.
arguments
    acquisition
    options.Plot (1,1) logical = true
    options.Duration (1,1) double {mustBeNonnegative,mustBeFinite} = 0
end
if isa(acquisition,'Simulink.SimulationOutput')
    signal = acquisition.get('ecgRaw');
else
    signal = acquisition;
end
filename = "";
if options.Duration > 0 && isa(signal,'timeseries') && numel(signal.Time) < 2
    warning('plotECGRecording:IncompleteRecording','No complete ECG recording was logged.');
    return
end
assert(isa(signal,'timeseries') && numel(signal.Time) >= 2, ...
    'Expected an ECG timeseries with at least two logged samples.');
time = double(signal.Time(:));
adcCounts = double(signal.Data(:));
assert(numel(time) == numel(adcCounts) && all(isfinite(time)) && ...
    all(diff(time) > 0), 'Expected one sample per increasing timestamp.');
sampleInterval = median(diff(time));
nominalSampleRateHz = 1/sampleInterval;
if options.Duration > 0
    tolerance = 16*eps(options.Duration);
    if abs(time(1)) > tolerance || time(end) + sampleInterval < options.Duration - tolerance
        warning('plotECGRecording:IncompleteRecording', ...
            'Run ended before a full %g-second recording was logged.',options.Duration);
        return
    end
    withinRecording = time < options.Duration - tolerance;
    time = time(withinRecording);
    adcCounts = adcCounts(withinRecording);
end
time = time-time(1);

folder = fullfile(fileparts(mfilename('fullpath')),'recordings');
if ~isfolder(folder)
    mkdir(folder);
end
stamp = string(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
filename = fullfile(folder,"ecg_"+stamp+".mat");
save(filename,"time","adcCounts","nominalSampleRateHz");

if options.Plot
    figure(Name="A0 ECG Recording");
    plot(time,adcCounts);
    xlabel("Logged time (s)");
    ylabel("Raw ADC counts");
    title("Arduino A0 ECG recording");
    grid on;
end
fprintf("Saved %d samples to %s\n",numel(time),filename);
minCounts = min(adcCounts);
maxCounts = max(adcCounts);
fprintf("A0 ADC min: %g, max: %g, peak-to-peak: %g counts.\n", ...
    minCounts,maxCounts,maxCounts-minCounts);
if maxCounts == minCounts
    warning('plotECGRecording:ConstantInput', ...
        'All logged A0 samples have the same ADC count (%g).',minCounts);
end
end
