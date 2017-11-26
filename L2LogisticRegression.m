clc;
clear;
%%

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
tempX = zeros(15000,784);
tempY = zeros(15000,1);
k = 1;
for i=1:10
    random=randperm(6000);
    for j=1:1500
        tempX(k,:) = dataX(random(j),:,i);
        tempY(k,:) = i-1;
        k = k+1;
    end
   
end

random = randperm(15000);
X = zeros(15000,784);
Y = zeros(15000,1);
for i=1:15000
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
testX = zeros(3000,784);
testY = zeros(3000,1);
k=1;
for i=1:10
    random=randperm(1000);
    for j=1:300
        testX(k,:) = dataX(random(j),:,i);
        testY(k,:) = i-1;
        k = k+1;
    end
   
end
%n1=15000, n2=3000, D =784
[n1, D] = size(X);
[n2, ~] = size(testX);
trainX = zeros(n1,D+1);
trainX(:,1)=1;
trainX(:,2:D+1)=X(:,:);
trainY = Y;
validationX =zeros(n2,D+1);
validationX(:,1)=1;
validationX(:,2:D+1)=testX(:,:);

validationY = testY;
%% notMNIST DATASET
% addpath('Dataset/');
% data = importdata('notMNIST_small.mat');
% images = data.images;
% Originallabels = data.labels;
% [a,b,c] = size(images);
% OriginalX = zeros(c,a*b+1);
% random=randperm(c);
% 
% l=2;
% OriginalX(:,1) = 1;
% for i=1:c
%     for j=1:a
%         for k=1:b
%             OriginalX(i,l) = images(j,k,i); 
%             l = l+1;
%         end
%     end
%     l=2;
% end
% 
% ratio=0.8;
% n1=fix(0.8*c);
% D=a*b;
% n2=c-n1;
% in=1;
% testX=zeros(n2,D+1);
% testlabels=zeros(n2,1);
% c=n1;
% X=zeros(c,D+1);
% labels=zeros(c,1);
% 
% for i=1:c
%     X(in,:)=OriginalX(random(i),:);
%     labels(in,1)=Originallabels(random(i),:);
%     in=in+1;
% end
% 
% in=1;
% for i=c+1:c+n2
%     testX(in,:)=OriginalX(random(i),:);
%     testlabels(in,1)=Originallabels(random(i),1);
%     in=in+1;
% end
% trainX = X;
% trainY = labels;
% validationX = testX;
% validationY = testlabels;
%%

%gradient descent : learning W for each pair of class
'gradient descent'
W=randn(D+1,1);
no_of_classes=10;
no_of_classifiers=(no_of_classes*(no_of_classes-1))/2;
D=D+1;

%W=randn(D,no_of_classifiers);
 W=zeros(D,no_of_classifiers);
 W(1,:)=0.02;
% W(2,:)=0.03;
% W(3,:)=0.04;

%W = -0.0003+(0.0006)*rand(D,no_of_classifiers);

alpha=0.5;

lambda =12000;

cur=0;
for x = 1 : no_of_classes
    x
for y = x+1 : no_of_classes
cur=cur+1;
Wcurrent=W(:,cur);

cnew=0;
for s=1:n1
if trainY(s,1)==x-1||trainY(s,1)==y-1
cnew=cnew+1;
end
end

Ycurrent=zeros(cnew,1);
Xcurrent=zeros(cnew,D);

index=1;
for s=1:n1
if trainY(s,1)==x-1||trainY(s,1)==y-1
Xcurrent(index,:) = trainX(s,:);
if trainY(s,1)==x-1
Ycurrent(index,1)=0;
else
Ycurrent(index,1)=1; 
end
index=index+1;
end
end


R=zeros(cnew,cnew);
f=zeros(cnew,1);

for iteration=1:6
   
f=Xcurrent*Wcurrent;
f=sigmf(f,[1 0]);


for i=1:cnew
    R(i,i)=f(i,1)*(1-f(i,1));
end



H=zeros(D,D);
reg=eye(D,D);
reg(1,1)=0;

H=transpose(Xcurrent)*R*Xcurrent;

H=((1/cnew)*H)+ ((lambda/cnew)*reg);

diff=zeros(cnew,1);

diff=transpose(Xcurrent)*(f-Ycurrent);

    Wtemp=Wcurrent;
    Wtemp(1,1)=0;
    diff=((1/cnew)*diff)+((lambda/cnew)*Wtemp);
    
Wcurrent=Wcurrent-(inv(H)*diff);

end
W(:,cur)=Wcurrent;
end
end

Output=zeros(no_of_classes,n2);


x=0;
for u = 1 : no_of_classes
    u
for v = u+1 : no_of_classes
x=x+1;

Wcur=W(:,x);

    f=validationX*Wcur;
    f=sigmf(f,[1 0]);
    f=round(f);

for t=1:n2
if f(t,1)==0
Output(u,t)=Output(u,t)+1;
else
Output(v,t)=Output(v,t)+1;
end
end

end
end

total=0;

labelMatrix=max(Output);

for i=1:n2  
    i
for j=1:no_of_classes
    if labelMatrix(1,i)==Output(j,i)
         label=j-1;
         break;
    end
end

if label==validationY(i,1)
    total=total+1;
end
end
    
    accuracy=(total/n2)*100;

accuracy
