clc;
clear;

%%
addpath('../Load data functions/');
addpath('../Dataset/');
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
numberoftestingexamples = 3000;
idx = zeros(10,1);
for i=1:10000
    dataX(idx(testY(i,1)+1)+1,:,testY(i,1)+1) = testX(i,:);
    idx(testY(i,1)+1) = idx(testY(i,1)+1)+1;
end
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

%%
[ predictedY ] = svm1(X, Y, testX);
error = classification_error_svm(testY, predictedY)
