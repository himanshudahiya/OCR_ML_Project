clc;
clear;
addpath('Dataset/');
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
testX=zeros(3744,a*b+1);
testlabels=zeros(3744,1);

c=14980;
X=zeros(c,a*b+1);
labels=zeros(c,1);

for i=1:c
    X(in,:)=OriginalX(random(i),:);
    labels(in,1)=Originallabels(random(i),:);
    in=in+1;
end

in=1;
for i=c+1:c+3744
    testX(in,:)=OriginalX(random(i),:);
    testlabels(in,1)=Originallabels(random(i),1);
    in=in+1;
end



%gradient descent : learning W for each pair of class
W=randn(a*b+1,1);
no_of_classes=10;
no_of_classifiers=(no_of_classes*(no_of_classes-1))/2;
D=a*b+1;

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
for y = x+1 : no_of_classes
cur=cur+1;
Wcurrent=W(:,cur);

cnew=0;
for s=1:c
if labels(s,1)==x-1||labels(s,1)==y-1
cnew=cnew+1;
end
end

Ycurrent=zeros(cnew,1);
Xcurrent=zeros(cnew,D);

index=1;
for s=1:c
if labels(s,1)==x-1||labels(s,1)==y-1
Xcurrent(index,:) = X(s,:);
if labels(s,1)==x-1
Ycurrent(index,1)=0;
else
Ycurrent(index,1)=1; 
end
index=index+1;
end
end

model = svmtrain(Xcurrent, Ycurrent);
t1 = toc;
% classification

% [predicted_label, accuracy, decision_values]=svmpredict(test_label, test_set, model);
% t2 = toc;
% disp(num2str(t1));
% disp(num2str(t2));

end
end

c=3744;

Output=zeros(no_of_classes,c);


x=0;
for u = 1 : no_of_classes
for v = u+1 : no_of_classes
x=x+1;

Wcur=W(:,x);

    f=testX*Wcur;
    f=sigmf(f,[1 0]);
    f=round(f);

for t=1:c
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

for i=1:c    
for j=1:no_of_classes
    if labelMatrix(1,i)==Output(j,i)
         label=j-1;
         break;
    end
end

if label==testlabels(i,1)
    total=total+1;
end
end
    
    accuracy=(total/c)*100;

accuracy
