clc;
clear;
addpath('Dataset/');
data = importdata('notMNIST_small.mat');
images = data.images;
labels = data.labels;
[a,b,c] = size(images);
X = zeros(c,a*b);
l=1;
for i=1:c
    for j=1:a
        for k=1:b
            X(i,l) = images(j,k,i);
            l = l+1;
        end
    end
    l=1;
end

%splitting into training and validation
iporder = randperm(c);
ratio = 0.99;

n1 = fix(ratio*c);
n2 = c - n1;

trainX = zeros(n1,a*b);
trainY = zeros(n1,1);
validationX = zeros(n2,a*b);
validationY = zeros(n2,1);
for i=1:n1
    trainX(i,:) = X(iporder(i),:);
    trainY(i,1) = labels(iporder(i),1);
end

for i=1:n2
    validationX(i,:) = X(iporder(n1+i),:);
    validationY(i,1) = labels(iporder(n1+i),1);
end
D = a*b;
%% using MNIST dataset
% addpath('Load data functions/');
% addpath('Dataset/');
% trainX = loadMNISTImages('train-images.idx3-ubyte');
% trainY = loadMNISTLabels('train-labels.idx1-ubyte');
% testX = loadMNISTImages('t10k-images.idx3-ubyte');
% testY = loadMNISTLabels('t10k-labels.idx1-ubyte');
% 
% trainX = trainX.';
% testX = testX.';
% %% taking training data
% dataX = zeros(6000,784,10);
% 
% idx = zeros(10,1);
% for i=1:60000
%     dataX(idx(trainY(i,1)+1,1)+1,:,trainY(i,1)+1) = trainX(i,:);
%     idx(trainY(i,1)+1,1) = idx(trainY(i,1)+1,1)+1;
% end
% tempX = zeros(15000,784);
% tempY = zeros(15000,1);
% k = 1;
% for i=1:10
%     random=randperm(6000);
%     for j=1:1500
%         tempX(k,:) = dataX(random(j),:,i);
%         tempY(k,:) = i-1;
%         k = k+1;
%     end
%    
% end
% 
% random = randperm(15000);
% X = zeros(15000,784);
% Y = zeros(15000,1);
% for i=1:15000
%     X(i,:) = tempX(random(i),:);
%     Y(i,:) = tempY(random(i),:);
% end
% 
% 
% %% taking test data
% 
% dataX = zeros(1000,784,10);
% 
% idx = zeros(10,1);
% for i=1:10000
%     dataX(idx(testY(i,1)+1)+1,:,testY(i,1)+1) = testX(i,:);
%     idx(testY(i,1)+1) = idx(testY(i,1)+1)+1;
% end
% testX = zeros(3000,784);
% testY = zeros(3000,1);
% k=1;
% for i=1:10
%     random=randperm(1000);
%     for j=1:300
%         testX(k,:) = dataX(random(j),:,i);
%         testY(k,:) = i-1;
%         k = k+1;
%     end
%    
% end
% 
% [n1, D] = size(X);
% [n2, ~] = size(testX);
% trainX = X;
% trainY = Y;
% validationX = testX;
% validationY = testY;
%% kNN
k = 3;
out = zeros(n2,1);
n1
D
n2
for i=1:n2
    i
    tempDistance = zeros(n1,1);
    for j=1:n1
        sum=0;
        for q=1:D
            sum = sum + (validationX(i,q)-trainX(j,q)).^2;
        end
        tempDistance(j,1) = sum;
    end
    ind = zeros(k,1);
    output=zeros(k,1);
    
    for q=1:k
        [val,ind(q)] = min(tempDistance(:,1));
        ind(q)
        tempDistance(ind(q)) = []; 
        output(q) = trainY(ind(q));
    end
    out(i)=mode(output);
end


%% error calculation
error = 0;
for i=1:n2
    if(out(i)~=validationY(i))
        error = error+1;
    end
end

error = error*100/n2