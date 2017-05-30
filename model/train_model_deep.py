from conv_net_model_deep import *
import os
import logging
import matplotlib.pyplot as plt

if __name__ == "__main__":
    predict_year = 2013
    
    # Create a coordinator
    config = Config()
    RUN_NAME = "run6__deeper__soybean/" # DON'T FORGET TO CHANGE THIS!!!
    config.save_path = os.path.expanduser('~/cs231n-satellite-images-models/' + RUN_NAME)

    assert(len(os.listdir(config.save_path)) <= 1)
    logging.basicConfig(filename=os.path.join(config.save_path, str(predict_year)+'.log'),level=logging.DEBUG)

    
    # load data to memory
    filename = os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean' + '.npz')
    content = np.load(filename)
    image_all = content['output_image']
    yield_all = content['output_yield']
    year_all = content['output_year']
    locations_all = content['output_locations']
    index_all = content['output_index']

    # delete broken images
    list_delete=[]
    for i in range(image_all.shape[0]):
        if np.sum(image_all[i,:,:,:])<=287:
            if year_all[i]<2016:
                list_delete.append(i)
    image_all=np.delete(image_all,list_delete,0)
    yield_all=np.delete(yield_all,list_delete,0)
    year_all = np.delete(year_all,list_delete, 0)
    locations_all = np.delete(locations_all, list_delete, 0)
    index_all = np.delete(index_all, list_delete, 0)

    # split into train and validate
    index_train = np.nonzero(year_all < predict_year)[0]
    index_validate = np.nonzero(year_all == predict_year)[0]
    print 'train size',index_train.shape[0]
    print 'validate size',index_validate.shape[0]

    # calc train image mean (for each band), and then detract (broadcast)
    image_mean=np.mean(image_all[index_train],(0,1,2))
    image_all = image_all - image_mean

    image_validate=image_all[index_validate]
    yield_validate=yield_all[index_validate]

    model= NeuralModel(config,'net')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)
    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.initialize_all_variables())

    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []

    train_loss=0
    val_loss=0
    val_prediction = 0
    val_deviation = np.zeros([config.B])
    
    #########################
    # block when test
    # add saver
    saver=tf.train.Saver()

    RMSE_min = 100
    try:
        for i in range(config.train_step):
            if i==3500:
                config.lr/=10
            if i==20000:
                config.lr/=10

            index_train_batch = np.random.choice(index_train,size=config.B)
            image_train_batch = image_all[index_train_batch,:,0:config.H,:]
            yield_train_batch = yield_all[index_train_batch]
            year_train_batch = year_all[index_train_batch,np.newaxis]

            index_validate_batch = np.random.choice(index_validate, size=config.B)

            _, train_loss = sess.run([model.train_op, model.loss_err], feed_dict={
                model.x:image_train_batch,
                model.y:yield_train_batch,
                model.lr:config.lr,
                model.keep_prob: config.drop_out
                })

            if i%200 == 0:
                val_loss,fc6,W,B = sess.run([model.loss_err,model.fc6,model.dense_W,model.dense_B], feed_dict={
                    model.x: image_all[index_validate_batch, :, 0:config.H, :],
                    model.y: yield_all[index_validate_batch],
                    model.keep_prob: 1
                })

                print 'predict year'+str(predict_year)+'step'+str(i),train_loss,val_loss,config.lr
                logging.info('predict year %d step %d %f %f %f',predict_year,i,train_loss,val_loss,config.lr)
            if i%200 == 0:
                # do validation
                pred = []
                real = []
                for j in range(image_validate.shape[0] / config.B):
                    real_temp = yield_validate[j * config.B:(j + 1) * config.B]
                    pred_temp= sess.run(model.logits, feed_dict={
                        model.x: image_validate[j * config.B:(j + 1) * config.B,:,0:config.H,:],
                        model.y: yield_validate[j * config.B:(j + 1) * config.B],
                        model.keep_prob: 1
                        })
                    pred.append(pred_temp)
                    real.append(real_temp)
                pred=np.concatenate(pred)
                real=np.concatenate(real)
                RMSE=np.sqrt(np.mean((pred-real)**2))
                ME=np.mean(pred-real)

                if RMSE<RMSE_min:
                    RMSE_min=RMSE
                    save_path = saver.save(sess, config.save_path + str(predict_year)+'CNN_model.ckpt')
                    print('save in file: %s' % save_path)
                    np.savez(config.save_path+str(predict_year)+'result.npz',
                        summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                        summary_RMSE=summary_RMSE,summary_ME=summary_RMSE)

                print 'Validation set','RMSE',RMSE,'ME',ME,'RMSE_min',RMSE_min
                logging.info('Validation set RMSE %f ME %f RMSE_min %f',RMSE,ME,RMSE_min)
            
                summary_train_loss.append(train_loss)
                summary_eval_loss.append(val_loss)
                summary_RMSE.append(RMSE)
                summary_ME.append(ME)
    except KeyboardInterrupt:
        print 'stopped'
    finally:
        # save
        save_path = saver.save(sess, config.save_path + str(predict_year)+'CNN_model.ckpt')
        print('save in file: %s' % save_path)
        logging.info('save in file: %s' % save_path)

        # save result
        pred_out = []
        real_out = []
        feature_out = []
        year_out = []
        locations_out =[]
        index_out = []
        for i in range(image_all.shape[0] / config.B):
            feature,pred = sess.run(
                [model.fc6,model.logits], feed_dict={
                model.x: image_all[i * config.B:(i + 1) * config.B,:,0:config.H,:],
                model.y: yield_all[i * config.B:(i + 1) * config.B],
                model.keep_prob:1
            })
            real = yield_all[i * config.B:(i + 1) * config.B]

            pred_out.append(pred)
            real_out.append(real)
            feature_out.append(feature)
            year_out.append(year_all[i * config.B:(i + 1) * config.B])
            locations_out.append(locations_all[i * config.B:(i + 1) * config.B])
            index_out.append(index_all[i * config.B:(i + 1) * config.B])
            
        weight_out, b_out = sess.run(
            [model.dense_W, model.dense_B], feed_dict={
                model.x: image_all[0 * config.B:(0 + 1) * config.B, :, 0:config.H, :],
                model.y: yield_all[0 * config.B:(0 + 1) * config.B],
                model.keep_prob: 1
            })

        print pred_out
        pred_out=np.concatenate(pred_out)
        real_out=np.concatenate(real_out)
        feature_out=np.concatenate(feature_out)
        year_out=np.concatenate(year_out)
        locations_out=np.concatenate(locations_out)
        index_out=np.concatenate(index_out)
        
        path = config.save_path + str(predict_year)+'result_prediction.npz'
        np.savez(path,
            pred_out=pred_out,real_out=real_out,feature_out=feature_out,
            year_out=year_out,locations_out=locations_out,weight_out=weight_out,b_out=b_out,index_out=index_out)

        np.savez(config.save_path+str(predict_year)+'result.npz',
                        summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                        summary_RMSE=summary_RMSE,summary_ME=summary_ME)
        # plot results
        npzfile = np.load(config.save_path+str(predict_year)+'result.npz')
        summary_train_loss=npzfile['summary_train_loss']
        summary_eval_loss=npzfile['summary_eval_loss']
        summary_RMSE = npzfile['summary_RMSE']
        summary_ME = npzfile['summary_ME']

        # Plot the points using matplotlib
        plt.plot(range(len(summary_train_loss)), summary_train_loss)
        plt.plot(range(len(summary_eval_loss)), summary_eval_loss)
        plt.xlabel('Training steps')
        plt.ylabel('L2 loss')
        plt.title('Loss curve')
        plt.legend(['Train', 'Validate'])
        plt.show()

        plt.plot(range(len(summary_RMSE)), summary_RMSE)
        # plt.plot(range(len(summary_ME)), summary_ME)
        plt.xlabel('Training steps')
        plt.ylabel('Error')
        plt.title('RMSE')
        # plt.legend(['RMSE', 'ME'])
        plt.show()

        # plt.plot(range(len(summary_RMSE)), summary_RMSE)
        plt.plot(range(len(summary_ME)), summary_ME)
        plt.xlabel('Training steps')
        plt.ylabel('Error')
        plt.title('ME')
        # plt.legend(['RMSE', 'ME'])
        plt.show()
