%% CMPT-741 template code for: sentiment analysis base on Convolutional Neural Network
% Author: Samaneh Khakshour, Pouya Fattahi
% Date: November 30, 2016
clc;
%clear all;
rng(1234);
%% Section 1: Preparation before training
%[data, wordMap] = read_dataall();
wordMap('<PAD>')=length(wordMap) +1;
wordMap('<UNK>')=length(wordMap) +1;

min_length=4;
train_length=length(data)*0.8;
valid_length=length(data)*0.2;

train=cell(length(train_length),2);
X_new=[];
for i=1:length(data)
    s=data{i,2};
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
    label=data{i,3};
    len_new=length(s);
     for k=1:len_new
         word_index(k)=wordMap(s{k});
     end
    train{i,1}=word_index;
    train{i,2}=label;
    word_index=[];
end
 
Train=train(1:train_length,:);
Valid=train(train_length+1:end,:);  

d=128;
total_words=length(wordMap);
T=normrnd(0,0.1,[total_words+1,d]);
filter_size=[2,3,4];
n_filter=128;

w=cell(length(filter_size),1);
B=cell(length(filter_size),1);

for i=1: length(filter_size)
    f=filter_size(i);
    w{i}=normrnd(0,0.1,[f,d,1,n_filter]);
    B{i}=zeros(n_filter,1);
end

total_filters=length(filter_size)*n_filter;
n_class=2;
w_out=normrnd(0,0.1,[1,1,total_filters,n_class]);
B_out=zeros(n_class,1);

step_size=1e-2;
reg=1e-3;


%% Section 2: Training
for iter=1:50
    predict=0;
    losstot=0;
    iter
for j=1:train_length 
    %  Forward propagation 
    word_index=Train{j,1};
    X=T(word_index,:);
    y=Train{j,2};
    if y==0
        y=1;
    else
        y=2;
    end
    pool_res=cell(1,length(filter_size));
    cache=cell(2,length(filter_size));
    for i=1:length(filter_size)
        conv=vl_nnconv(single(X),single(w{i}),single(B{i}));
        relu=vl_nnrelu(conv);    
        sizes=size(relu);
        pool=vl_nnpool(relu,[sizes(1),1]);  
        cache{2,i}=relu;
        cache{1,i}=conv;
        pool_res{i}=pool;
    end
    z=vl_nnconcat(pool_res,3);
    [ydrop,mask] = vl_nndropout(z);
    o=vl_nnconv(ydrop,single(w_out),single(B_out));  
    [~,pred]=max(o);
    if pred==y
        predict=predict+1;
    end
    loss=vl_nnloss(o,y);
    losstot=losstot+loss;
    %output layer back propagation
    s_o=size(o);
    dzdy = 1 ;
    % Back-propagation
    dzdx_loss = vl_nnloss(o, y, dzdy);
    [dzdx_conv2l, dzdw_out_conv2, dzdb_out_conv2] = vl_nnconv(ydrop,single(w_out),single(B_out), dzdx_loss) ;
    dzdx_conv2 = vl_nndropout(z, dzdx_conv2l, 'mask', mask);
    dzdx_conv2_1=dzdx_conv2(:,:,1:d*1);
    dzdx_conv2_2=dzdx_conv2(:,:,d*1+1:d*2);
    dzdx_conv2_3=dzdx_conv2(:,:,d*2+1:d*3);
    %1 is for the filter h=2
    %2 is for the filter h=3
    %3 is for the filter h=4
    
    %Back propagate Pooling
    sizes=size(cache{2,1});
    dzdx_pool1_1=vl_nnpool(cache{2,1},[sizes(1),1],dzdx_conv2_1);
    sizes=size(cache{2,2});
    dzdx_pool1_2=vl_nnpool(cache{2,2},[sizes(1),1],dzdx_conv2_2);
    sizes=size(cache{2,3});
    dzdx_pool1_3=vl_nnpool(cache{2,3},[sizes(1),1],dzdx_conv2_3);
    % Back propagate Relu
    dzdx_relu1_1=vl_nnrelu(cache{1,1}, dzdx_pool1_1);
    dzdx_relu1_2=vl_nnrelu(cache{1,2}, dzdx_pool1_2);
    dzdx_relu1_3=vl_nnrelu(cache{1,3}, dzdx_pool1_3);
    %Back propagate CNN
    [dxdy_conv1_1, dxdw_conv1_1, dxdb_conv1_1] = vl_nnconv(single(X),single(w{1}),single(B{1}), dzdx_relu1_1);
    [dxdy_conv1_2, dxdw_conv1_2, dxdb_conv1_2] = vl_nnconv(single(X),single(w{2}),single(B{2}), dzdx_relu1_2);
    [dxdy_conv1_3, dxdw_conv1_3, dxdb_conv1_3] = vl_nnconv(single(X),single(w{3}),single(B{3}), dzdx_relu1_3);
    %Updating wwights and biases  
    trainwrate=-0.01;
    trainBrate=-0.001;
    trainwrate2=-0.01;
    trainBrate2=-0.001;
    w{1}=w{1}+trainwrate*dxdw_conv1_1;
    w{2}=w{2}+trainwrate*dxdw_conv1_2;
    w{3}=w{3}+trainwrate*dxdw_conv1_3;
    B{1}=B{1}+trainBrate*dxdb_conv1_1;
    B{2}=B{2}+trainBrate*dxdb_conv1_2;
    B{3}=B{3}+trainBrate*dxdb_conv1_3;  
    w_out=w_out+trainwrate2*dzdw_out_conv2;
    B_out=B_out+trainBrate2*dzdb_out_conv2;
end 
predict
end
%% Validation
predictV=0;
losstotV=0;
for j=1:valid_length 
    word_index=Valid{j,1};
    XV=T(word_index,:);
    yV=Valid{j,2};
    if yV==0
        yV=1;
    else
        yV=2;
    end
    pool_res=cell(1,length(filter_size));
    cache=cell(2,length(filter_size));
    for i=1:length(filter_size)
        conv=vl_nnconv(single(XV),single(w{i}),single(B{i}));
        relu=vl_nnrelu(conv);
        sizes=size(relu);
        pool=vl_nnpool(relu,[sizes(1),1]);
        cache{2,i}=relu;
        cache{1,i}=conv;
        pool_res{i}=pool;
    end
    z=vl_nnconcat(pool_res,3);
    [ydrop,mask] = vl_nndropout(z);
    o=vl_nnconv(z,single(w_out),single(B_out));
    [~,pred]=max(o);
    if pred==yV
        predictV=predictV+1;
    end
    loss=vl_nnloss(o,yV);
    losstotV=losstotV+loss;
end
accuracy=predictV/valid_length


