function [data, wordMap] = read_dataall()
% CMPT-741 example code for: reading training data and building vocabulary.
% NOTE: reading testing data is similar, but no need to build the vocabulary.
%
% return: 
%       data(cell), 1st column -> sentence id, 2nd column -> words, 3rd column -> label
%       wordMap(Map), contains all words and their index, get word index by calling wordMap(word)

%% File 1: Given text file

headLine = true;
separater = '::';

words = [];

data = cell(6000+6921+1822+873, 3);  %with  original data + STS2

fid = fopen('C:\Users\Sam\Desktop\CMPT 741\project\DataMining\train.txt', 'r');
line = fgets(fid);

ind = 1;
while ischar(line)
    if headLine
        line = fgets(fid);
        headLine = false;
    end
    attrs = strsplit(line, separater);
    sid = str2double(attrs{1});
    
    s = attrs{2};
    w = strsplit(s);
    words = [words w];
    
    y = str2double(attrs{3});
    
    % save data
    data{ind, 1} = sid;
    data{ind, 2} = w;
    data{ind, 3} = y;
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end
%% File 2: stsabinary 1
fid = fopen('C:\Users\Sam\Desktop\CMPT 741\project\DataMining\stsabinary1.txt', 'r');
line = fgets(fid);

while ischar(line)
    s = line(3:end);
    w = strsplit(s);
    words = [words w];
    
    y = str2double(line(1));
    data{ind, 1} = ind;
    data{ind, 2} = w;
    data{ind, 3} = y;
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end
%% File 3: stsabinary 2
fid = fopen('C:\Users\Sam\Desktop\CMPT 741\project\DataMining\stsabinary2.txt', 'r');
line = fgets(fid);

while ischar(line)
    s = line(3:end);
    w = strsplit(s);
    words = [words w];
    
    y = str2double(line(1));
    data{ind, 1} = ind;
    data{ind, 2} = w;
    data{ind, 3} = y;
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end
%% File 4: stsabinary 3
fid = fopen('C:\Users\Sam\Desktop\CMPT 741\project\DataMining\stsabinary3.txt', 'r');
line = fgets(fid);

while ischar(line)
    s = line(3:end);
    w = strsplit(s);
    words = [words w];
    
    y = str2double(line(1));
    data{ind, 1} = ind;
    data{ind, 2} = w;
    data{ind, 3} = y;
    
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end

% %% File 5: MR Data
% fid = fopen('C:\Users\yroshan\Desktop\samaneh\TextProject\mrdata.txt', 'r');
% line = fgets(fid);
% 
% while ischar(line)
%     s = line(3:end);
%     w = strsplit(s);
%     words = [words w];
%     
%     y = str2double(line(1));
%     data{ind, 1} = ind;
%     data{ind, 2} = w;
%     data{ind, 3} = y;
%     
%     % read next line
%     line = fgets(fid);
%     ind = ind + 1;
% end

%% Making Wordmap
words = unique(words);
wordMap = containers.Map(words, 1:length(words));
fprintf('finish loading data and vocabulary\n');