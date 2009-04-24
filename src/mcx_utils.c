#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mcx_utils.h"

void mcx_savedata(float *dat,int len,Config *cfg){
     FILE *fp;
     char name[MAX_PATH_LENGTH];
     sprintf(name,"%s.mc2",cfg->session);
     fp=fopen(name,"wb");
     fwrite(dat,sizeof(float),len,fp);
     fclose(fp);
}

void mcx_normalize(float field[], float scale, int fieldlen){
     int i;
     for(i=0;i<fieldlen;i++){
         field[i]*=scale;
     }
}

void mcx_error(int id,char *msg){
     fprintf(stderr,"MCX ERROR(%d):%s\n",id,msg);
     exit(id);
}

void mcx_readconfig(char *fname, Config *cfg){
     if(fname[0]==0){
     	mcx_loadconfig(stdin,cfg);
        if(cfg->session[0]=='\0'){
		strcpy(cfg->session,"default");
	}
     }
     else{
     	FILE *fp=fopen(fname,"rt");
	if(fp==NULL) mcx_error(-2,"can not load specified config file");
	mcx_loadconfig(fp,cfg); 
	fclose(fp);
        if(cfg->session[0]=='\0'){
		strcpy(cfg->session,fname);
	}
     }
}

void mcx_writeconfig(char *fname, Config *cfg){
     if(fname[0]==0)
     	mcx_saveconfig(stdout,cfg);
     else{
     	FILE *fp=fopen(fname,"wt");
	if(fp==NULL) mcx_error(-2,"can not load specified config file");
	mcx_saveconfig(fp,cfg);     
	fclose(fp);
     }
}

void mcx_initcfg(Config *cfg){
     cfg->medianum=0;
     cfg->detnum=0;
     cfg->dim.x=0;
     cfg->dim.y=0;
     cfg->dim.z=0;
     cfg->totalmove=0;
     cfg->nthread=0;
     cfg->seed=0;
     cfg->isrowmajor=1; /* default is C array*/
     cfg->maxgate=1;
     cfg->isreflect=1;
     cfg->isnormalized=0;
     cfg->issavedet=1;
     cfg->respin=1;
     cfg->issave2pt=1;

     cfg->prop=NULL;
     cfg->detpos=NULL;
     cfg->vol=NULL;
     cfg->session[0]='\0';
}

void mcx_clearcfg(Config *cfg){
     if(cfg->medianum)
     	free(cfg->prop);
     if(cfg->detnum)
     	free(cfg->detpos);
     if(cfg->dim.x && cfg->dim.y && cfg->dim.z)
        free(cfg->vol);

     mcx_initcfg(cfg);
}

void mcx_loadconfig(FILE *in, Config *cfg){
     int i,gates;
     char filename[MAX_PATH_LENGTH]={0}, comment[MAX_PATH_LENGTH];
     
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of photons: [1000000]\n\t");
     fscanf(in,"%d", &(cfg->nphoton) ); 
     fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d\nPlease specify the random number generator seed: [1234567]\n\t",cfg->nphoton);
     fscanf(in,"%d", &(cfg->seed) );
     fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d\nPlease specify the position of the source: [10 10 5]\n\t",cfg->seed);
     fscanf(in,"%f %f %f", &(cfg->srcpos.x),&(cfg->srcpos.y),&(cfg->srcpos.z) );
     fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the normal direction of the source fiber: [0 0 1]\n\t",
                                   cfg->srcpos.x,cfg->srcpos.y,cfg->srcpos.z);
     cfg->srcpos.x--;cfg->srcpos.y--;cfg->srcpos.z--; /*convert to C index*/
     fscanf(in,"%f %f %f", &(cfg->srcdir.x),&(cfg->srcdir.y),&(cfg->srcdir.z) );
     fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the time gates in seconds (start end and step) [0.0 1e-9 1e-10]\n\t",
                                   cfg->srcdir.x,cfg->srcdir.y,cfg->srcdir.z);
     fscanf(in,"%f %f %f", &(cfg->tstart),&(cfg->tend),&(cfg->tstep) );
     fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %f %f\nPlease specify the path to the volume binary file:\n\t",
                                   cfg->tstart,cfg->tend,cfg->tstep);
     if(cfg->tstart>cfg->tend || cfg->tstep==0.f){
         mcx_error(-9,"incorrect time gate settings");
     }
     gates=int((cfg->tend-cfg->tstart)/cfg->tstep+0.5);
     if(cfg->maxgate>gates)
	 cfg->maxgate=gates;

     fscanf(in,"%s", filename);
     fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%s\nPlease specify the x voxel size (in mm), x dimension, min and max x-index [1.0 100 1 100]:\n\t",filename);
     fscanf(in,"%f %d %d %d", &(cfg->steps.x),&(cfg->dim.x),&(cfg->crop0.x),&(cfg->crop1.x));
     fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the y voxel size (in mm), y dimension, min and max y-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.x,cfg->dim.x,cfg->crop0.x,cfg->crop1.x);
     fscanf(in,"%f %d %d %d", &(cfg->steps.y),&(cfg->dim.y),&(cfg->crop0.y),&(cfg->crop1.y));
     fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the z voxel size (in mm), z dimension, min and max z-index [1.0 100 1 100]:\n\t",
                                  cfg->steps.y,cfg->dim.y,cfg->crop0.y,cfg->crop1.y);
     fscanf(in,"%f %d %d %d", &(cfg->steps.z),&(cfg->dim.z),&(cfg->crop0.z),&(cfg->crop1.z));
     fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%f %d %d %d\nPlease specify the total types of media:\n\t",
                                  cfg->steps.z,cfg->dim.z,cfg->crop0.z,cfg->crop1.z);
     fscanf(in,"%d", &(cfg->medianum));
     cfg->medianum++;
     fgets(comment,MAX_PATH_LENGTH,in);

     if(in==stdin)
     	fprintf(stdout,"%d\n",cfg->medianum);
     cfg->prop=(Medium*)malloc(sizeof(Medium)*cfg->medianum);
     cfg->prop[0].mua=0.f; /*property 0 is already air*/
     cfg->prop[0].mus=0.f;
     cfg->prop[0].g=0.f;
     cfg->prop[0].n=1.f;
     for(i=1;i<cfg->medianum;i++){
        if(in==stdin)
		fprintf(stdout,"Please define medium #%d: mus(1/mm), anisotropy, mua(1/mm) and refractive index: [1.01 0.01 0.04 1.37]\n\t",i);
     	fscanf(in, "%f %f %f %f", &(cfg->prop[i].mus),&(cfg->prop[i].g),&(cfg->prop[i].mua),&(cfg->prop[i].n));
        fgets(comment,MAX_PATH_LENGTH,in);
        if(in==stdin)
		fprintf(stdout,"%f %f %f %f\n",cfg->prop[i].mus,cfg->prop[i].g,cfg->prop[i].mua,cfg->prop[i].n);
     }
     if(in==stdin)
     	fprintf(stdout,"Please specify the total number of detectors and fiber diameter (in mm):\n\t");
     fscanf(in,"%d %f", &(cfg->detnum), &(cfg->detradius));
     fgets(comment,MAX_PATH_LENGTH,in);
     if(in==stdin)
     	fprintf(stdout,"%d %f\n",cfg->detnum,cfg->detradius);
     cfg->detpos=(float4*)malloc(sizeof(float4)*cfg->detnum);
     cfg->issavedet=(cfg->detpos>0);
     for(i=0;i<cfg->detnum;i++){
        if(in==stdin)
		fprintf(stdout,"Please define detector #%d: x,y,z (in mm): [5 5 5 1]\n\t",i);
     	fscanf(in, "%f %f %f", &(cfg->detpos[i].x),&(cfg->detpos[i].y),&(cfg->detpos[i].z));
        cfg->detpos[i].x--;cfg->detpos[i].y--;cfg->detpos[i].z--;  /*convert to C index*/
        fgets(comment,MAX_PATH_LENGTH,in);
        if(in==stdin)
		fprintf(stdout,"%f %f %f\n",cfg->detpos[i].x,cfg->detpos[i].y,cfg->detpos[i].z);
     }
     if(filename[0]){
        mcx_loadvolume(filename,cfg);
     }else{
     	mcx_error(-4,"one must specify a binary volume file in order to run the simulation");
     }
}

void mcx_saveconfig(FILE *out, Config *cfg){
     int i;

     fprintf(out,"%d\n", (cfg->nphoton) ); 
     fprintf(out,"%d\n", (cfg->seed) );
     fprintf(out,"%f %f %f\n", (cfg->srcpos.x),(cfg->srcpos.y),(cfg->srcpos.z) );
     fprintf(out,"%f %f %f\n", (cfg->srcdir.x),(cfg->srcdir.y),(cfg->srcdir.z) );
     fprintf(out,"%f %f %f\n", (cfg->tstart),(cfg->tend),(cfg->tstep) );
     fprintf(out,"%f %d %d %d\n", (cfg->steps.x),(cfg->dim.x),(cfg->crop0.x),(cfg->crop1.x));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.y),(cfg->dim.y),(cfg->crop0.y),(cfg->crop1.y));
     fprintf(out,"%f %d %d %d\n", (cfg->steps.z),(cfg->dim.z),(cfg->crop0.z),(cfg->crop1.z));
     fprintf(out,"%d", (cfg->medianum));
     for(i=0;i<cfg->medianum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->prop[i].mus),(cfg->prop[i].g),(cfg->prop[i].mua),(cfg->prop[i].n));
     }
     fprintf(out,"%d", (cfg->detnum));
     for(i=0;i<cfg->detnum;i++){
     	fprintf(out, "%f %f %f %f\n", (cfg->detpos[i].x),(cfg->detpos[i].y),(cfg->detpos[i].z),(cfg->detpos[i].w));
     }
}

void mcx_loadvolume(char *filename,Config *cfg){
     int datalen,res;
     FILE *fp=fopen(filename,"rb");
     if(fp==NULL){
     	     mcx_error(-5,"specified binary volume file does not exist");
     }
     if(cfg->vol){
     	     free(cfg->vol);
     	     cfg->vol=NULL;
     }
     datalen=cfg->dim.x*cfg->dim.y*cfg->dim.z;
     cfg->vol=(unsigned char*)malloc(sizeof(unsigned char)*datalen);
     res=fread(cfg->vol,sizeof(unsigned char),datalen,fp);
     fclose(fp);
     if(res!=datalen){
     	 mcx_error(-6,"file size does not match specified dimensions");
     }
}

void mcx_parsecmd(int argc, char* argv[], Config *cfg){
     int i=1,isinteractive=1;
     char filename[MAX_PATH_LENGTH]={0};
     
     if(argc<=1){
     	mcx_usage(argv[0]);
     	exit(0);
     }
     while(i<argc){
     	    if(argv[i][0]=='-'){
	        switch(argv[i][1]){
		     case 'i':
		     		isinteractive=1;
				break;
		     case 'f': 
		     		isinteractive=0;
		     	        if(i<argc-1)
				    strcpy(filename,argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
				break;
		     case 'm':
		     	        if(i<argc-1)
		     		    cfg->totalmove=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 't':
		     	        if(i<argc-1)
		     		    cfg->nthread=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 's':
		     	        if(i<argc-1)
		     		    strcpy(cfg->session,argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 'a':
		     	        if(i<argc-1)
		     		    cfg->isrowmajor=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 'g':
		     	        if(i<argc-1)
		     		    cfg->maxgate=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 'b':
		     	        if(i<argc-1)
		     		    cfg->isreflect=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 'd':
		     	        if(i<argc-1)
		     		    cfg->issavedet=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 'r':
		     	        if(i<argc-1)
		     		    cfg->respin=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 'S':
		     	        if(i<argc-1)
		     		    cfg->issave2pt=atoi(argv[++i]);
				else
				    mcx_error(-1,"incomplete input");
		     	        break;
		     case 'U':
 	                        cfg->isnormalized=1;
		     	        break;
		}
	    }
	    i++;
     }
     if(isinteractive){
          mcx_readconfig("",cfg);
     }else{
     	  mcx_readconfig(filename,cfg);
     }
}

void mcx_usage(char *exename){
     printf("\
#######################################################################################\n\
#                     Monte-Carlo Extreme (MCX) -- CUDA                               #\n\
#             Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>                    #\n\
#                                                                                     #\n\
#      Martinos Center for Biomedical Imaging, Massachusetts General Hospital         #\n\
#######################################################################################\n\
usage: %s <param1> <param2> ...\n\
where possible parameters include (the first item in [] is the default value)\n\
     -i 	   interactive mode\n\
     -f config     read config from a file\n\
     -m n_move	   total move (integration intervals) per thread\n\
     -t [1|int]    total thread number\n\
     -r [1|int]    number of re-spins (repeations)\n\
     -a [1|0]      1 for C array, 0 for Matlab array\n\
     -d [1|0]      1 to save photon info at detectors, 0 not to save\n\
     -g [1|int]    number of time gates per run\n\
     -b [1|0]      1 to bounce the photons at the boundary, 0 to exit\n\
     -s sessionid  a string to identify this specific simulation\n\
     -S [1|0]      1 to save the fluence field, 0 do not save\n\
     -U            normailze the fluence to unitary\n\
example:\n\
       %s -t 1024 -m 100000 -f input.inp -s test -d 0\n",exename,exename);
}
