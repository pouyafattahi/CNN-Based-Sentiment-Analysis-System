%% Read File

headLine = true;
separater = '::';

%words = [];
datatest = cell(1000, 4); 

fid = fopen('C:\Users\Sam\Desktop\CMPT 741\project\DataMining\sample_test.txt', 'r');
% fid = fopen('C:\Users\yroshan\Desktop\samaneh\TextProject\train.txt', 'r');
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
    wo = strsplit(s);
%     resul=str2double(attrs{3});
    % save data
    datatest{ind, 1} = sid;
    datatest{ind, 2} = wo;
%     datatest{ind,4}=resul;
    % read next line
    line = fgets(fid);
    ind = ind + 1;
end

%% Preparing Data
min_length=4;
test_length=length(datatest);
test=cell(length(test_length),1);
for i=1:length(datatest)
    s=datatest{i,2};
    len=length(s);
    if len < min_length
        new_s= cell(1,min_length);
        for j=1:min_length
            if j<=min_length-len
                new_s{j}='<PAD>';
            else
                new_s{j}=s{j-min_length+len};
            end
        end
        s=new_s;
    end

    len_new=length(s);
     for k=1:len_new
         if isKey(wordMap,s{k})
             word_index(k)=wordMap(s{k});
         else
             wordMap(s{k})=length(wordMap) +1;
             word_index(k)=wordMap(s{k});
         end
     end
   test{i,1}=word_index;
    word_index=[];
end
TestSet=test;
total_words=length(wordMap);
if (total_words > max(size(T)))
    T = [T;normrnd(0,0.1,[total_words-max(size(T)),d])];
end


%% Testing
fid = fopen('C:\Users\Sam\Desktop\CMPT 741\project\DataMining\sample_eval.txt', 'w');
formatSpec = '%d::%d\r\n';
corrects=0;
for j=1:test_length 
% for j=1:train_length 
    word_index=TestSet{j,1};
%        word_index=Train{j,1};  %%REMOVE THIS LINE..THIS IS JUST FOR TESTING
%        yV=Train{j,2};  %%REMOVE THIS LINE..THIS IS JUST FOR TESTING
    XV=T(word_index,:);
    pool_res_test=cell(1,length(filter_size));
    for i=1:length(filter_size)
        conv=vl_nnconv(single(XV),single(w{i}),single(B{i}));
        relu=vl_nnrelu(conv);
    
        sizes=size(relu);
        pool=vl_nnpool(relu,[sizes(1),1]);

        pool_res_test{i}=pool;
    end
    z=vl_nnconcat(pool_res_test,3);
    [ydrop,mask] = vl_nndropout(z);
    o=vl_nnconv(z,single(w_out),single(B_out));
    
    [~,pred]=max(o);
    datatest{j,3}=pred-1;   % To make it 0 and 1
    
    %% Printing to file
%     if datatest{j,3}==datatest{j,4}
%         corrects=corrects+1;
%     end
%     if (yV==(pred-1))
%         corrects=corrects+1;
%     end
   fprintf(fid,formatSpec,datatest{j,1},datatest{j,3});
end


 