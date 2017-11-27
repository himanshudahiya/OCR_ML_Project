clc;
clear;
%%
data = importdata('notMNIST_small.mat');
images = data.images;
Originallabels = data.labels;
[a,b,c] = size(images);
OriginalX = zeros(c,a*b+1);
random=randperm(c);

l=2;
OriginalX(:,1) = 1;
for i=1:c
    for j=1:a
        for k=1:b
            OriginalX(i,l) = images(j,k,i); 
            l = l+1;
        end
    end
    l=2;
end

in=1;
ratio = 0.8;
n1 = fix(ratio*c);
n2 = c-n1;
testX=zeros(n2,a*b+1);
testY=zeros(n2,1);


c = n1;
X=zeros(c,a*b+1);
Y=zeros(c,1);

for i=1:c
    X(in,:)=OriginalX(random(i),:);
    Y(in,1)=Originallabels(random(i),:);
    in=in+1;
end

Means=mean(X(:,2:a*b+1));
standardDeviation=std(X(:,2:a*b+1));

for i=1:c
    X(i,2:a*b+1)=X(i,2:a*b+1)-Means;
end

for j=1:(a*b)
    X(:,j+1)=X(:,j+1)/standardDeviation(j);
end

in=1;
for i=c+1:c+n2
    testX(in,:)=OriginalX(random(i),:);
    testY(in,1)=Originallabels(random(i),1);
    in=in+1;
end

Means2=mean(testX(:,2:a*b+1));
standardDeviation2=std(testX(:,2:a*b+1));

for i=1:n2
    testX(i,2:a*b+1)=testX(i,2:a*b+1)-Means2;
end

for j=1:(a*b)
    testX(:,j+1)=testX(:,j+1)/standardDeviation2(j);
end

%%
[ predictedY ] = svm1(X, Y, testX);
error = classification_error_svm(testY, predictedY)
