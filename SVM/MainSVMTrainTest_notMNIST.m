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

ratio = 0.80;
n1 = fix(ratio*c);
n2 = c-n1;
in=1;
testX=zeros(n2,a*b+1);
testY=zeros(n2,1);

X=zeros(n1,a*b+1);
Y=zeros(n1,1);

for i=1:n1
    X(in,:)=OriginalX(random(i),:);
    Y(in,1)=Originallabels(random(i),:);
    in=in+1;
end

in=1;
for i=n1+1:n1+n2
    testX(in,:)=OriginalX(random(i),:);
    testY(in,1)=Originallabels(random(i),1);
    in=in+1;
end

%%
[ predictedY ] = svm1(X, Y, testX);
error = classification_error_svm(testY, predictedY)
