clc;
clear;
%% using MNIST dataset
addpath('Load data functions/');
addpath('Dataset/');
trainX = loadMNISTImages('train-images.idx3-ubyte');
trainY = loadMNISTLabels('train-labels.idx1-ubyte');
testX = loadMNISTImages('t10k-images.idx3-ubyte');
testY = loadMNISTLabels('t10k-labels.idx1-ubyte');

trainX = trainX.';
testX = testX.';
%% taking training data
dataX = zeros(6000,784,10);

idx = zeros(10,1);
for i=1:60000
    dataX(idx(trainY(i,1)+1,1)+1,:,trainY(i,1)+1) = trainX(i,:);
    idx(trainY(i,1)+1,1) = idx(trainY(i,1)+1,1)+1;
end
numberoftrainingexamples = 15000;
tempX = zeros(numberoftrainingexamples,784);
tempY = zeros(numberoftrainingexamples,1);
k = 1;
for i=1:10
    random=randperm(6000);
    for j=1:numberoftrainingexamples/10
        tempX(k,:) = dataX(random(j),:,i);
        tempY(k,:) = i-1;
        k = k+1;
    end
   
end

random = randperm(numberoftrainingexamples);
X = zeros(numberoftrainingexamples,784);
Y = zeros(numberoftrainingexamples,1);
for i=1:numberoftrainingexamples
    X(i,:) = tempX(random(i),:);
    Y(i,:) = tempY(random(i),:);
end


%% taking test data

dataX = zeros(1000,784,10);

idx = zeros(10,1);
for i=1:10000
    dataX(idx(testY(i,1)+1)+1,:,testY(i,1)+1) = testX(i,:);
    idx(testY(i,1)+1) = idx(testY(i,1)+1)+1;
end
numberoftestingexamples = 3000;
testX = zeros(numberoftestingexamples,784);
testY = zeros(numberoftestingexamples,1);
k=1;
for i=1:10
    random=randperm(1000);
    for j=1:numberoftestingexamples/10
        testX(k,:) = dataX(random(j),:,i);
        testY(k,:) = i-1;
        k = k+1;
    end
   
end

[n1, D] = size(X);
[n2, ~] = size(testX);
trainX = X;
trainY = Y;
testingX = testX;
testingY = testY;

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
            sum = sum + (testingX(i,q)-trainX(j,q)).^2;
        end
        tempDistance(j,1) = sum;
    end
    ind = zeros(k,1);
    output=zeros(k,1);
    
    for q=1:k
        [val,ind(q)] = min(tempDistance(:,1));
        ind(q);
        tempDistance(ind(q)) = []; 
        output(q) = trainY(ind(q));
    end
    out(i)=mode(output);
end


%% error calculation
error = 0;
for i=1:n2
    if(out(i)~=testingY(i))
        error = error+1;
    end
end

error = error*100/n2