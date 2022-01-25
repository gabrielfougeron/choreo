# import Choreo_target_custom_test as cho

# nmul=500
# for imul in range(nmul):
    # epsmul = imul*1./100.
    # cho.main(epsmul=epsmul)
    
    
    
import Choreo_find

all_kwargs = Choreo_find.main()

Choreo_find.Find_Choreo(**all_kwargs)
