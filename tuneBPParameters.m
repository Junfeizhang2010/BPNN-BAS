function wb = tuneBPParameters(data, neuronCount, tuneMode, opt)
    %% BP�������ʼ�����Ż�
    % ���룺
    %   data             - ���ݼ�
    %   neuronCount      - ÿ����Ԫ����
    %   tuneMode         - �Ż��㷨ѡ��'BAS'����ţ���㷨
    %                                   ''���㷨
    %   ifDraw           - �Ƿ�ͼ
    % �����
    %   para             - ���������Ż��㷨�Ƽ���BP������Ȩ�غ�ƫ��
    %                    ��һ�����ز�ʱ�ṹ��
    %                                 ��������������������������������������������������������
    %           ��               ��     ���ز���Ԫ1                ��������ֵ1
    %           ��          neuronCount ���ز���Ԫ2                ��������ֵ1
    %           ��               ��        ����
    %           ��               ��     ���ز���ԪneureCount       ��������ֵ1
    %           ��               ��     ���ز���Ԫ1                ��������ֵ2
    %           ��          neuronCount ���ز���Ԫ2                ��������ֵ2
    %           ��               ��        ����
    %           ��               ��     ���ز���ԪneureCount       ��������ֵ2
    %   size(trainX,1)��neuronCount                           ����
    %           ��               ��     ���ز���Ԫ1                ��������ֵsize(trainX,1)
    %           ��          neuronCount ���ز���Ԫ2                ��������ֵsize(trainX,1)
    %           ��               ��        ����
    %           ��               ��     ���ز���ԪneureCount       ��������ֵsize(trainX,1)
    %
    %           ��               ��     ����Ŀ��ֵ1                  ���ز���Ԫ1
    %           ��          neuronCount ����Ŀ��ֵ1                  ���ز���Ԫ2
    %           ��               ��         ����
    %           ��               ��     ����Ŀ��ֵ1                  ���ز���ԪneureCount
    %           ��               ��     ����Ŀ��ֵ2                  ���ز���Ԫ1
    %           ��          neuronCount ����Ŀ��ֵ2                  ���ز���Ԫ2
    %           ��               ��         ����
    %           ��               ��     ����Ŀ��ֵ2                  ���ز���ԪneureCount
    %   size(trainY,1)��neuronCount         ����
    %           ��               ��     ����Ŀ��ֵsize(trainY,1)     ���ز���Ԫ1
    %           ��          neuronCount ����Ŀ��ֵsize(trainY,1)     ���ز���Ԫ2
    %           ��               ��         ����
    %           ��               ��     ����Ŀ��ֵsize(trainY,1)     ���ز���ԪneureCount
    %
    %                            ��     ���ز���Ԫ1ƫ��
    %                       neuronCount ���ز���Ԫ2ƫ��
    %                            ��        ����
    %                            ��     ���ز���ԪneureCountƫ��
    %                            ��     �������Ԫ1ƫ��
    %                   size(trainY,1)  �������Ԫ2ƫ��
    %                            ��        ����
    %                            ��     �������Ԫsize(trainY,1)ƫ��
    %                                  ��������������������������������������������������������
    %                    ��������ز�ʱ�ṹ��
    %                                 ��������������������������������������������������������
    %           ��               ��     ���ز�1��Ԫ1                ��������ֵ1
    %           ��       neuronCount(1) ���ز�1��Ԫ2                ��������ֵ1
    %           ��               ��        ����
    %           ��               ��     ���ز�1��ԪneureCount(1)    ��������ֵ1
    %           ��               ��     ���ز�1��Ԫ1                ��������ֵ2
    %           ��       neuronCount(1) ���ز�1��Ԫ2                ��������ֵ2
    %           ��               ��        ����
    %           ��               ��     ���ز�1��ԪneureCount(1)    ��������ֵ2
    %  size(trainX,1) �� neuronCount(1)                         ����
    %           ��               ��     ���ز�1��Ԫ1                ��������ֵsize(trainX,1)
    %           ��       neuronCount(1) ���ز�1��Ԫ2                ��������ֵsize(trainX,1)
    %           ��               ��        ����
    %           ��               ��     ���ز�1��ԪneureCount(1)    ��������ֵsize(trainX,1)
    %
    %           ��               ��     ���ز�2��Ԫ1                ���ز�1��Ԫ1
    %           ��       neuronCount(2) ���ز�2��Ԫ2                ���ز�1��Ԫ2
    %           ��               ��        ����
    %           ��               ��     ���ز�2��ԪneureCount(2)    ���ز�1��ԪneureCount(1)
    %           ��               ��     ���ز�2��Ԫ1                ���ز�1��Ԫ1
    %           ��       neuronCount(2) ���ز�2��Ԫ2                ���ز�1��Ԫ2
    %           ��               ��        ����
    %           ��               ��     ���ز�2��ԪneureCount(2)    ���ز�1��ԪneureCount(1)
    %  size(trainX,1) �� neuronCount(2)                         ����
    %           ��               ��     ���ز�2��Ԫ1                ���ز�1��Ԫ1
    %           ��       neuronCount(2) ���ز�2��Ԫ2                ���ز�1��Ԫ2
    %           ��               ��        ����
    %           ��               ��     ���ز�2��ԪneureCount(2)    ���ز�1��ԪneureCount(1)
    %
    %           ��               ��     ����Ŀ��ֵ1                  ���ز�2��Ԫ1
    %           ��          neuronCount ����Ŀ��ֵ1                  ���ز�2��Ԫ2
    %           ��               ��         ����
    %           ��               ��     ����Ŀ��ֵ1                  ���ز�2��ԪneureCount(2)
    %           ��               ��     ����Ŀ��ֵ2                  ���ز�2��Ԫ1
    %           ��          neuronCount ����Ŀ��ֵ2                  ���ز�2��Ԫ2
    %           ��               ��         ����
    %           ��               ��     ����Ŀ��ֵ2                  ���ز�2��ԪneureCount(2)
    %   size(trainY,1)��neuronCount()         ����
    %           ��               ��     ����Ŀ��ֵsize(trainY,1)     ���ز�2��Ԫ1
    %           ��          neuronCount ����Ŀ��ֵsize(trainY,1)     ���ز�2��Ԫ2
    %           ��               ��         ����
    %           ��               ��     ����Ŀ��ֵsize(trainY,1)     ���ز�2��ԪneureCount(2)
    %
    %                            ��     ���ز�1��Ԫ1ƫ��
    %                    neuronCount(1) ���ز�1��Ԫ2ƫ��
    %                            ��        ����
    %                            ��     ���ز�1��ԪneureCount(1)ƫ��
    %                            ��     ���ز�2��Ԫ1ƫ��
    %                    neuronCount(1) ���ز�2��Ԫ2ƫ��
    %                            ��        ����
    %                            ��     ���ز�2��ԪneureCount(2)ƫ��
    %                            ��     �������Ԫ1ƫ��
    %                   size(trainY,1)  �������Ԫ2ƫ��
    %                            ��        ����
    %                            ��     �������Ԫsize(trainY,1)ƫ��
    %                                  ��������������������������������������������������������
    
    algoMode = 'BP';
    fprintf(['            �����ڽ���' algoMode 'ģ�ͳ�ʼ�����Ż���']);
    switch tuneMode
    %% ��ţ���㷨
    case 'BAS'
        disp(['ʹ��' tuneMode '�㷨���е���...']);
        opt.eta     = 0.9; % ���ñ䲽����Խ�ӽ�Ŀ��Խ��
        opt.c       = 5;    % cΪ�������������ı�����ʼ�չ̶��������Ų�����̣��������Ҳ��̡��������Ƽ�5��
        opt.step    = 1;    % ���ó�ʼ�����������ܴ�������Ա�����󳤶��൱
        opt.n       = 100;   % ����������
        switch size(neuronCount, 2)
            case 1
                k = size(data.crossColumn{1}.trainX,1) * neuronCount + neuronCount +...
                    size(data.crossColumn{1}.trainY,1) * neuronCount + size(data.crossColumn{1}.trainY,1);
                % ά�� = ����ֵ�� * ���ز���Ԫ���� + ���ز���Ԫ���� + Ŀ��ֵ�� * ���ز���Ԫ���� + Ŀ��ֵ��
            case 2
                k = size(data.crossColumn{1}.trainX,1) * neuronCount(1) + neuronCount(1) +...
                    neuronCount(1) * neuronCount(2) + neuronCount(2) +...
                    size(data.crossColumn{1}.trainY,1) * neuronCount(2) + size(data.crossColumn{1}.trainY,1);
                % ά�� = ����ֵ�� * ���ز�1��Ԫ���� + ���ز�1��Ԫ���� + ���ز�1��Ԫ���� * ���ز�2��Ԫ���� +
                %        ���ز�2��Ԫ���� * Ŀ��ֵ�� + Ŀ��ֵ��
        end
        
        x = rands(k,1);   % xΪ��ţλ�ã������ʼ��Ϊ-1��1������������[ά�ȡ�1]
        others.neuronCount = neuronCount;
        [xStore, fBestStore] = BAS(data, 'BPParameter', x, opt, others);
        
        % �ó�����
        [~, bestIndex] = min(fBestStore);
        wb = xStore(2 : end - 1, bestIndex);
        
        
        % ��ͼ
        if ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
            % ���ս�����ڵ���������ͼ
            figurePara = setFigurePara(data.dataIndex, data.posOutput);
            figurePara.label.x = 'Iteration';
            figurePara.label.y = figurePara.label.mse;
            figurePara.label.title = ['Convergence line of ' algoMode ' with different parameters while ' tuneMode ' tuning'];
            figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
            figurePara.preserve.name = ['ʹ��' tuneMode '�㷨�Ż�' algoMode 'Ȩ�غ�ƫ�õ����ս�����ڵ���������ͼ'];
            drawSingleLine(1 : size(fBestStore, 1), fBestStore, figurePara);
        end
        disp(['              ' algoMode 'ģ�ͳ�ʼ�����Ż���ϡ�']);
        
    %% �㷨2
    case ''
        
    end