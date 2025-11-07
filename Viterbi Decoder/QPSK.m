%%


clear 

clc

tic
n = 2000 ; % number of input symbols

fs = 10 ; % sampling frequency 

T = 1 ; % g(t) pulse duration

%%%%%%%%%% Creating filter transfer functions

t_g = 0:1/fs:T-1/fs ; % sampling points

g_T = 1/sqrt(T) * ones(1, length(t_g)); % transmitter pulse

filt_g_T = filt(g_T , 1 ,1/fs) ; %transmitter pulse shaping filter


t_c = 0:1/fs:2*T-1/fs ; % sampling points

channel_func = @(t) sqrt(3/(2*T))*(1-t/(2*T)) ;
channel_coeffs = channel_func(t_c) ; % channel filter coeffs

filt_channel = filt(channel_coeffs , 1 ,1/fs);

temp1 = filt_g_T*filt_channel;
temp2 = temp1.Numerator ;  % building matched filter
temp3 = flip(temp2{1,1})/fs; % by flipping convolution of transmitter and channel coeffs

filt_g_R = filt(temp3 , 1 , 1/fs);


%%%%%%%%%%% Generating Random Sequence
%rng(2);
inp1 = 2*(randi(2,1,n)-1)-1 ; % sequence of random 1 , -1
inp2 = 2*(randi(2,1,n)-1)-1 ;

inp = 1/sqrt(2)*inp1 + 1i/sqrt(2)*inp2;

impulse = [1 , zeros(1, fs-1)] ;

seq = [] ;
for i = 1:n
    seq = [seq , inp(i)*impulse];
end


%%%%%%%%%%% manipulating the transmitted signal to create received signal

a = lsim(filt_g_T,seq,(0:length(seq)-1)/fs);

b = lsim(filt_channel,a,(0:length(seq)-1)/fs)/fs;

x_pulse = lsim(filt_g_T*filt_channel*filt_g_R ,[impulse , zeros(1, (6-1)*length(t_g)-4+1)],(0:(6)*length(t_g)-4+1-1)/fs)/(fs^2) ;

L = floor(length(x_pulse)/fs/2) ;
%%%% noise power at Yn is (|H(f)|^2)*N0 for Ni and Nq each,
%%%signal power is 1*X(0)^2 so SNR is X(0)^2/(2N0*|H(f)|^2)

N1 = conv(channel_coeffs,g_T)/fs;
N2 = sum(N1.^2)/fs;

S1 = x_pulse((L+1)*fs-1)^2;
SNR = -20:2:20;

N0 = S1/(2*N2)./(10.^(SNR/10)) ; 

BER=[];
for t = 1:length(N0)

mynoise = sqrt(N0(t)).*randn(2,length(b));

%Received_signal = awgn(b, -10,'measured')' ;

Received_signal = b' + mynoise(1,:) + 1i*mynoise(2,:);
noise = mynoise(1,:) + 1i*mynoise(2,:);

c1 = (conv(noise,temp3)/fs)';
c1 = c1(1:length(b));
c = (conv(Received_signal,temp3)/fs)';

c = c(1:length(b));

index = 1:length(b);
samples = downsample(circshift(index',2),fs) ;

sampled = c(samples) ; % each impulse extends to fs*(T+2T+(2T+T))-(number of filters)+1 samples, distance of sampling is T*fs samples.

%  if fs == 100 , then signal extends to 597, and the peak is 299th sample

% the first L+1 samples are dropped after downsampling due to delay of filters

y = sampled(L+2:end);

% because of limited data and 6L forward data requirement, not all the bits
% could be estimated

%%%%%%%%%%%%  Viterbi decoder

detected = [];
cost_reg = zeros(4,1);
path_reg = zeros(4,6*L);

for i=1:length(y)
    
    vit = binary_cost_func (y(i),cost_reg,path_reg,L,x_pulse,fs,i);
    [path_reg,cost_reg] =  path_cost_adjust(path_reg,vit,4);
    
           [~,I] = max(cost_reg);
           
    if  i == length(y)
       detected = [detected , path_reg(I,:)]; 
       break;
    end
    if i>=6*L
       detected = [detected , path_reg(I,1)]; 
    end
    

    
end

r = sum(inp(1:length(detected))==detected)/length(detected);

BER = [BER ,1-r];

end

plot(SNR,BER);
title('QPSK');
xlabel('SNR dB');
ylabel('Bit Error Rate');

toc
%% FUNCTIONs

function y = binary_cost_func(yn,cost,path,L,filt_pulse,fs,j) 

y11 = cost(1) + real(conj(1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)+1i/sqrt(2)))) ;
y12 = cost(1) + real(conj(1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)-1i/sqrt(2)))) ;
y13 = cost(1) + real(conj(-1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)+1i/sqrt(2)))) ;
y14 = cost(1) + real(conj(-1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)-1i/sqrt(2)))) ;

y21 = cost(2) + real(conj(1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)+1i/sqrt(2)))) ;
y22 = cost(2) + real(conj(1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)-1i/sqrt(2)))) ;
y23 = cost(2) + real(conj(-1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)+1i/sqrt(2)))) ;
y24 = cost(2) + real(conj(-1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)-1i/sqrt(2)))) ;

y31 = cost(3) + real(conj(1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)+1i/sqrt(2)))) ;
y32 = cost(3) + real(conj(1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)-1i/sqrt(2)))) ;
y33 = cost(3) + real(conj(-1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)+1i/sqrt(2)))) ;
y34 = cost(3) + real(conj(-1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)-1i/sqrt(2)))) ;

y41 = cost(4) + real(conj(1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)+1i/sqrt(2)))) ;
y42 = cost(4) + real(conj(1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(1/sqrt(2)-1i/sqrt(2)))) ;
y43 = cost(4) + real(conj(-1/sqrt(2)+1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)+1i/sqrt(2)))) ;
y44 = cost(4) + real(conj(-1/sqrt(2)-1i/sqrt(2))*(2*yn - filt_pulse((L+1)*fs-1)*(-1/sqrt(2)-1i/sqrt(2)))) ;


    for i =1:L
      
        y11 = y11 + real(conj(1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(1,end-i+1))) ;
        y12 = y12 + real(conj(1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(1,end-i+1))) ;
        y13=  y13 + real(conj(-1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(1,end-i+1))) ;
        y14 = y14 + real(conj(-1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(1,end-i+1))) ;
        
        y21 = y21 + real(conj(1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(2,end-i+1))) ;
        y22 = y22 + real(conj(1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(2,end-i+1))) ;
        y23=  y23 + real(conj(-1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(2,end-i+1))) ;
        y24 = y24 + real(conj(-1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(2,end-i+1))) ;
        
        y31 = y31 + real(conj(1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(3,end-i+1))) ;
        y32 = y32 + real(conj(1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(3,end-i+1))) ;
        y33=  y33 + real(conj(-1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(3,end-i+1))) ;
        y34 = y34 + real(conj(-1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(3,end-i+1))) ;
        
        
        y41 = y41 + real(conj(1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(4,end-i+1))) ;
        y42 = y42 + real(conj(1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(4,end-i+1))) ;
        y43=  y43 + real(conj(-1/sqrt(2)+1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(4,end-i+1))) ;
        y44 = y44 + real(conj(-1/sqrt(2)-1i/sqrt(2))*(-2 * filt_pulse((L+1+i)*fs-1)*path(4,end-i+1))) ;
        
    end
    
        [a1,b1] = max([y11,y21,y31,y41]);
        [a2,b2] = max([y12,y22,y32,y42]);   
        [a3,b3] = max([y13,y23,y33,y43]);
        [a4,b4] = max([y14,y24,y34,y44]); 
        

        
        
        y = [a1,b1; a2,b2; a3,b3; a4,b4]  ;
        
        if j == 1
           y = [a1,1;a2,2;a3,3;a4,4]  ; 
        end
        
        
end

 

function y=qpsk_map(a)
if a == 1
    y = 1/sqrt(2)+1i/sqrt(2) ;
end
if a == 2
    y = 1/sqrt(2)-1i/sqrt(2) ;
end
if a == 3
    y = -1/sqrt(2)+1i/sqrt(2) ;
end
if a == 4
    y = -1/sqrt(2)-1i/sqrt(2) ;
end
end

function [y_path,y_cost] = path_cost_adjust(path,vit,M)
k = zeros(1,M);
v = vit(:,2) ;
b = vit(:,1) ;
for i = 1 : M
    k(v(i)) = k(v(i)) + 1; % number of presence
end
u = k;
for i = 1 : M
if k(i)~=0 
    o = [] ;
    h = find(v==i);
    temp_path = [];
    temp_cost = [];
    
        for j = 1:k(i)
           o = path(i,:);
           o = circshift(o,-1);
           o(1 ,end) = qpsk_map(h(j));
           temp_cost = [temp_cost;vit(h(j),1)];
           temp_path = [temp_path ; o];
        end
        b(i) = temp_cost(1);
        path(i,:) = temp_path(1,:);
        w = find(u==0);
    if k(i)>1
        for j = 2:k(i)
           b(w(j-1)) = temp_cost(j);
           path(w(j-1),:)=temp_path(j,:);
           u(w(j-1))=1;
        end
    end
end   
    
end

y_path = path ;
y_cost = b ;

end