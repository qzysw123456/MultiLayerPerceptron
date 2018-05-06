train = load('optdigits_train.txt');
valid = load('optdigits_valid.txt');
test = load('optdigits_test.txt');

m = 15;
k = 10;
d = size(train,2) - 1;

w = (rand(m,d+1)-0.5)/50;
v = (rand(k,m+1)-0.5)/50;

[z w v] = mlptrain(train,valid,m,k);


Z = zeros(length(train)+length(valid),m);
for i = 1 : length(train)
    X = [train(i,1:d) 1];
    for j = 1 : m
        Z(i,j) = w(j,:) *  X';
    end
end

for i = (1 + length(train)) : (length(valid) + length(train))
    X = [valid(i-length(train),1:d) 1];
    for j = 1 : m
        Z(i,j) = w(j,:) *  X';
    end
end

Z2 = zeros(length(test),m);
for i = 1 : length(test)
    X = [test(i,1:d) 1];
    for j = 1 : m
        Z2(i,j) = w(j,:) *  X';
    end
end

coeff = pca(Z);

A = coeff(:,1:3);

project = Z * A;
project2 = Z2 * A;

scatter(project(:,1),project(:,2),10,[train(:,d+1);valid(:,d+1)],'fill');
xlabel('dimension-1');
ylabel('dimension-2');
title('plot of first 2 dimensions');

scatter3(project(:,1),project(:,2),project(:,3),10,[train(:,d+1);valid(:,d+1)],'fill');
xlabel('dimension-1');
ylabel('dimension-2');
zlabel('dimension-3');
title('plot of first 3 dimensions');