load("training2017/A00001.mat");
fs = 300;
t = (0:numel(val)-1)/fs;
plot(t, val);
xlabel('Time (s)');
ylabel('Signal');
grid on;
