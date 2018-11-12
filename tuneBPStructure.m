function [neuronCount, fBestCompared] = tuneBPStructure(data, tuneMode, opt)
    %% BP������ṹ�Ż�
    % ���룺
    %   data             - ���ݼ�
    %   tuneMode         - �Ż��㷨ѡ��'BAS'����ţ���㷨
    %   opt              - �㷨����
    % �����
    %   neuronCount      - ���������Ż��㷨�Ƽ���ÿ����Ԫ����
    %   fBestCompared    - ��������ͼ����
    
    algoMode = 'BP';
    fprintf(['            �����ڽ���' algoMode 'ģ�ͽṹ�Ż���']);
    if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'hiddenLevelMax'), opt.BP.hLMaxCount = 3; end; % �����������
    
    %% ����
    switch tuneMode
    % ��ţ���㷨
    case 'BAS'
        disp(['ʹ��' tuneMode '�㷨���е���...']);
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'eta'),  opt.BP.BAS.eta  = 0.95; end; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'c'),    opt.BP.BAS.c    = 5;    end; % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'step'), opt.BP.BAS.step = 10;   end; % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'BAS') || ~isfield(opt.BP.BAS, 'n'),    opt.BP.BAS.n    = 50;   end; % ����������

        % ����1-hLMaxCount����������
        if ~isfield(opt, 'BP') || ~isfield(opt.BP, 'init')
            for k = 1 : opt.BP.hLMaxCount
                if k == 1
                    opt.BP.init{k} = 30;
                elseif k == 2
                    opt.BP.init{k} = [20 10]';
                elseif k == 3
                    opt.BP.init{k} = [20 10 10]';
                elseif k == 4
                    opt.BP.init{k} = [10 10 10 10]';
                else
                    opt.BP.init{k} = randperm(40,k)'; 
                end
            end
        end
        for k = 1 : opt.BP.hLMaxCount
            disp(['                  ���ڲ����������Ϊ' num2str(k) '/' num2str(opt.BP.hLMaxCount) '�����...']);
            
            % ����ţ���㷨���㲻ͬ�����µ�������Ԫ����
            if ~isfield(opt, 'tune') || ~isfield(opt.tune, 'time')
                opt.BP.BAS.time = 10; 
            else
                opt.BP.BAS.time = opt.tune.time;
            end
            crossTime = size(data.crossColumn, 2);
            [xStore{k}, fBestStore{k}] = BAS(data, 'BPStructure', opt.BP.init{k}, opt.BP.BAS, []);
            [fBestBest(k), bestIndex(k)] = min(fBestStore{k});
            xBest{k} = xStore{k}(2 : end - 1, bestIndex(k));
        end
        
        % �ó�����
        [fBestBestForAll, hiddenLevelCount] = min(fBestBest);
        neuronCount = xBest{hiddenLevelCount}';
        disp(['              ' algoMode 'ģ�ͽṹ�Ż���ϣ���ţ���㷨�Ƽ�'...
            num2str(hiddenLevelCount) '�����ز㣬ÿ����Ԫ�ֱ�Ϊ' num2str(neuronCount) '��']);

    % �㷨2
    case ''
            
    end
    
    %% ��ͼ
    if ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        fBestCompared = fBestStore{hiddenLevelCount};
        n = size(fBestCompared, 1);
        % ���ս�����ڵ���������ͼ
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.x = 'Iteration';
        figurePara.label.y = figurePara.label.mse;
        if hiddenLevelCount == 1
            figurePara.label.title = ['Convergence line of ' algoMode ' with ' num2str(hiddenLevelCount)...
                ' hidden level and different number of neurons while' tuneMode 'tuning'];
        else
            figurePara.label.title = ['Convergence line of ' algoMode ' with ' num2str(hiddenLevelCount)...
                ' hidden levels and different number of neurons while' tuneMode 'tuning'];
        end
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = [data.dataIndex 'ʹ��' tuneMode '�㷨�Ż�' algoMode 'ģ�ͽṹ�����ս�����ڵ���������ͼ'];
        drawSingleLine(1 : n, fBestCompared, figurePara);

        % ���������Ŀ�꺯��ֵͼ
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.x = 'Iteration';
        figurePara.label.y = figurePara.label.mse;
        figurePara.label.title = ['Convergence line of ' algoMode ' with different number of hidden levels while ' tuneMode ' tuning'];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż�' algoMode 'ģ�ͽṹ����������'];
        drawBPLevelsTune(fBestStore, figurePara);
    end