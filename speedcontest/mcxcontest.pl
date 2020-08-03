#!/usr/bin/perl
###############################################################################
#
# MCX Benchmark and Speed Contest
#
# Author:  Qianqian Fang <q.fang at neu.edu>
# License: GPLv3
# Version: 0.5
# URL:     http://mcx.space/gpubench
# Github:  https://github.com/fangq/mcx/
#
###############################################################################

use strict;
use warnings;

my $MCX="../bin/mcx";
my $URL="http://mcx.space/gpubench/";
my $POSTURL="http://mcx.space/gpubench/gpucontest.cgi";

my %report=();
my %jsonopt=(utf8 => 1, pretty => 1);

# parse commandline options
my $options='';
my $mcxopt='-n 1e8';
while(my $opt = $ARGV[0]) {
    if($opt =~ /^-[ocsl-]/){
        if($opt eq '-c'){
		$jsonopt{'pretty'}=0;
	}elsif($opt eq '-s'){
		$options.='s';
	}elsif($opt eq '-l'){
		$options.='l';
	}elsif($opt eq '-o'){
		shift;
		$mcxopt=shift;
		next;
	}elsif($opt eq '--bin'){
		shift;
		$MCX=shift;
		next;
	}elsif($opt eq '--get'){
		shift;
		$URL=shift;
		next;
	}elsif($opt eq '--post'){
		shift;
		$POSTURL=shift;
		next;
	}elsif($opt eq '--help'){
		printf("%s - running MCX benchmarks and submit to MCX Speed Contest
	Format: %s <option1> <option2> ...\n
The supported options include (multiple parameters can be used, separated by spaces)
	-s      submit result to MCX Speed Contest by default (will ask if not used)
	-l      compare your benchmark with other GPUs submitted by other users
	-c      print JSON in compact form, otherwise, print in the indented form
	-o 'mcx options'   supply additional mcx command line options, such as '-n 1e6'
	--bin /path/to/mcx  manually specify mcx binary location (default: ../bin/mcx)
	--get url  specify mcx benchmark web link (default: http://mcx.space/gpubench)
	--post url specify submission link (default: http://mcx.space/gpubench/gpucontest.cgi)\n",
		$0,$0);
		exit 0;
	}else{
		die('invalid command line option');
	}
	shift;
    }else{
        die('invalid command line option');
    }
}

my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
$report{'date'} = sprintf("%.4d-%.2d-%.2d %.2d:%.2d:%.2d", $year+1900, $mon+1, $mday, $hour, $min, $sec);

## perform benchmark

&listgpu(\%report,'gpu',$mcxopt);
&runbench('cube60',\%report,'benchmark1',17.69,$mcxopt);
&runbench('cube60b',\%report,'benchmark2',27.24,$mcxopt);
&runbench('cube60planar',\%report,'benchmark3',25.51,$mcxopt);

## assemble report data

$report{'speedsum'}=$report{'benchmark1'}{'speed'}+$report{'benchmark2'}{'speed'}
    +$report{'benchmark3'}{'speed'};
$report{'version'}=1;
$report{'mcxversion'}=$report{'benchmark1'}{'mcxversion'};

printf("Speed test completed\nYour device(s) score is %.0f\n",$report{'speedsum'});

## download report and compare

my $ans='';

if(! ($options=~/l/)){
	print "Do you want to compare your score with other submitted results ([yes]/no)?\n";
	$ans=<STDIN>;
	chomp $ans; 
}
if($options=~/l/ || $ans =~/^yes/i || $ans eq ''){
	&comparegpu($URL);
}
$ans='';

## submit report
if(! ($options=~/s/)){
	print "\nDo you wish to submit your GPU benchmark data to our database (yes/[no])?\n";
	$ans=<STDIN>;
	chomp $ans; 
}
if($options=~/s/ || $ans =~/^yes/i){
	&submitresult(\%report,$POSTURL,\%jsonopt);
}
$ans='';

#==============================================================================
sub listgpu(){
	my ($report, $keyname, $mcxopt)=@_;
	my $mcxoutput=`$MCX -L`;

	$mcxoutput=~s/\e\[\d+(?>(;\d+)*)m//g;

	printf("Running '$MCX -L' to inquire GPUs ...\n");

	if($mcxoutput =~ /Global Memory/){
		my @gpustr=split(/=+\s+GPU Infomation\s+=+/,$mcxoutput);
		foreach my $gpurec (@gpustr){
			my @gpudata=split(/\n/,$gpurec);
			my %gpuinfo=();
			foreach my $info (@gpudata){
				if($info =~ /Device (\d+) of (\d+)\s*:\s*(.*)$/){
					$gpuinfo{'name'}=$3;
					$gpuinfo{'id'}=$1+0;
					$gpuinfo{'devcount'}=$2+0;
				}elsif($info =~ /Compute Capability\s*:\s*(\d+)\.(\d+)/){
					$gpuinfo{'major'}=$1+0;
					$gpuinfo{'minor'}=$2+0;
				}elsif($info =~ /Global Memory\s*:\s*(\d+)\s*B/){
					$gpuinfo{'globalmem'}=$1+0;
				}elsif($info =~ /Constant Memory\s*:\s*(\d+)\s*B/){
					$gpuinfo{'constmem'}=$1+0;
				}elsif($info =~ /Shared Memory\s*:\s*(\d+)\s*B/){
					$gpuinfo{'sharedmem'}=$1+0;
				}elsif($info =~ /Registers\s*:\s*(\d+)$/){
					$gpuinfo{'regcount'}=$1+0;
				}elsif($info =~ /Clock Speed\s*:\s*([0-9.]+)\s+GHz/){
					$gpuinfo{'clock'}=$1*1e6;
				}elsif($info =~ /Number of Cores\s*:\s*(\d+)$/){
					$gpuinfo{'core'}=$1+0;
				}elsif($info =~ /Number of SM.*\s*:\s*(\d+)$/){
					$gpuinfo{'sm'}=$1+0;
				}elsif($info =~ /Auto-thread\s*:\s*(\d+)$/){
					$gpuinfo{'autothread'}=$1+0;
				}elsif($info =~ /Auto-block\s*:\s*(\d+)$/){
					$gpuinfo{'autoblock'}=$1+0;
				}
			}
			if(%gpuinfo){
				push(@{$report{$keyname}}, \%gpuinfo);
				if(!($mcxopt=~/-G\s+["']*[0-9]+['"]*/ || $mcxopt=~/--gpu\s+['"]*[0-9]+['"]*/)){
					last;
				}
			}
		}
		if(length($report{$keyname})==0){
			die('no GPU found on your system, can not continue');
		}
	}else{
                die('GPU listing failed, output incomplete');
        }
	if($mcxopt=~/-G\s+['"]*([0-9]+)['"]*/ || $mcxopt=~/--gpu\s+['"]*([0-9]+)['"]*/){
		if($1 ne '' && $1>0){
			my @newgpu=();
			my $mask=$1;
			my $count=0;
			if($mask =~/^[01]+$/){
				foreach my $i (split(undef,$mask)){
					if($count>=@{$report{$keyname}}){
						last;
					}
					push(@newgpu,$report{$keyname}[$count])  if($i==1);
					$count++;
				}
			}elsif($mask < @{$report{$keyname}}){
				push(@newgpu,$report{$keyname}[$mask-1]);
			}
			@{$report{$keyname}} = @newgpu;
		}
	}
}

#==============================================================================

sub runbench(){
	my ($benchname, $report, $keyname, $absorb, $mcxopt)=@_;

	my $mcxoutput=`$MCX --bench $benchname $mcxopt`;
	$mcxoutput=~s/\e\[\d+(?>(;\d+)*)m//g;

	printf("Running benchmark '$MCX --bench $benchname $mcxopt' ...\n");

	if($mcxoutput =~ /total simulated energy:/){
		my @lines=split(/\n/,$mcxoutput);
		my %bench=();
		my $tstart=0;
		foreach my $line (@lines){
			if($line =~ /np=(d+)\s+nthread=/){
				$bench{'stat'}{'nphoton'}=$1;
			}elsif($line =~ /simulation speed:\s*([-0-9.]+)/){
				$bench{'speed'}=$1+0;
			}elsif($line =~ /total simulated energy:\s*([-0-9.]+)\s*absorbed: ([-0-9.]+)%/){
				$bench{'stat'}{'energytot'}=$1+0;
				$bench{'stat'}{'absorbfrac'}=$2+0;
				$bench{'stat'}{'energyabs'}=$1*0.01*$2;
			}elsif($line =~ /normalization factor alpha=([-0-9.]+)/){
				$bench{'stat'}{'normalizer'}=$1+0;
			}elsif($line =~ /init complete\s*:\s*([0-9.]+)\s+ms/){
				$tstart=$1;
			}elsif($line =~ /kernel complete:\s+([0-9.]+)\s+ms/){
				$bench{'stat'}{'runtime'}=$1-$tstart;
			}elsif($line =~ /\$Rev::.*(\$Date::[^\$]+\$)/){
				$bench{'mcxversion'}=$1;
			}
		}
		if(%bench){
			$report{$keyname}=\%bench;
			my $absfrac=$bench{'stat'}{'absorbfrac'};
			if(abs($absfrac - $absorb)/$absorb > 0.1){
				die('$benchname failed (expected $absorb, got $absfrac), can not continue');
			}
		}else{
			die('$benchname failed, output incomplete');
		}
	}else{
                die('$benchname failed, output incomplete');
	}
}

#==============================================================================

sub comparegpu(){
	use LWP::Simple qw(get);
	use Term::ANSIColor qw(:constants);

	my ($url)=@_;
	my $database = get $url;
	if(!defined($database)){
		$database = `curl --silent $url`;
		die("fail to download database from $url") if($database eq '');
	}
	my @res=split(/\n/,$database);
	my $pos=0;
	for my $line (@res){
		if($line =~/<span class='score'>(\d+)</ && $report{'speedsum'}>= $1){
			last;
		}
		$pos++;
	}
	my $wid=`tput cols`;
	if($wid eq ''){
	    $wid=80;
	}

	my $maxscore=0;
	foreach my $id (($pos-5)..($pos+4)){
		next if($id<0 || $id>$#res);
		my $linewid=$wid-37;
		if($id==$pos){
			printf(RED);
			printf("%-25s:%8.0f [%s]\n", '>> Your GPU <<',$report{'speedsum'},'#' x int($linewid*($report{'speedsum'}/$maxscore)));
			printf(RESET);
		}
		if($res[$id]=~/<td>(.*)<\/td><td>\/\s*(.*)\s*\/<br\/><span class='hostid'>(.*)<\/span><\/td><td><div class='barchart'><span class='score'>(\d+)<\/span>/){
			$maxscore=($4>$maxscore)?$4:$maxscore;
			printf("%-25s:%8.0f [%s]\n",substr($2,0,25),$4,'#' x int($linewid*($4/$maxscore)));
		}
	}
}

#==============================================================================

sub submitresult(){
	use LWP::UserAgent ();
	use JSON;

	my ($report, $url, $jsonopt)=@_;
	my %form=();
	my %userinfo=(
	  "name"=>"Please provide your full name (will not publish)",
	  "email"=>"Please provide your email (will not publish)[required]",
	  "institution"=>"What is your institution/company (will not publish)?",
	  "nickname"=>"Please provide your web name/ID (visible on webpage)[required]",
	  "machine"=>"Short description of your computer (OS/version/driver, visible)[required]",
	  "comment"=>"Optional comment"
	);
	my @formitem=('name','email','institution','nickname','machine','comment');
	foreach my $key (@formitem){
		print $userinfo{$key}."\n\t";
		my $ans=<STDIN>;
		chomp $ans;
		if(($key eq 'nickname' || $key eq 'email' ||  $key eq 'machine' ) && $ans eq ''){
			while($ans eq ''){
				print "required field, can not be empty, please input again\n\t";
				$ans=<STDIN>;
				chomp $ans;
			}
		}
		$report{'userinfo'}{$key}=$ans;
	}
	$form{'name'}=$report{'userinfo'}{'nickname'};
	$form{'time'}= time();
	$form{'ver'}=substr($report{'mcxversion'},7,100);
	$form{'ver'}=~s/\$//g;
	$form{'b1'}=$report{'benchmark1'}{'speed'};
	$form{'b2'}=$report{'benchmark2'}{'speed'};
	$form{'b3'}=$report{'benchmark3'}{'speed'};
	$form{'score'}=$report{'speedsum'};
	$form{'computer'}=$report{'userinfo'}{'machine'}.":/";
	foreach my $gpu (@{$report{'gpu'}}){
		my $gpuname=%{$gpu}{'name'};
		$gpuname=~s/\s*GeForce\s*//g;
		$gpuname=~s/\s*NVIDIA\s*//g;
		$form{'computer'}.=$gpuname."/";
	}
	$form{'report'}= encode_json($report);

	print "Do you want to see the full data to be submitted ([yes]/no)?";
	my $ans=<STDIN>;
	chomp $ans;
	if($ans =~/^yes/i || $ans eq ''){
		print "---------------- begin data -----------------\n";
		print encode_json(\%form);
		print "----------------- end data ------------------\n";
	}
        $ans='';

	print "\nFinal confirmation - do you want ot submit ([yes]/no)?";
	$ans=<STDIN>;
	chomp $ans;
	if($ans =~/^yes/i || $ans eq ''){
		my $ua      = LWP::UserAgent->new(); 
		my $response = $ua->post( $url, \%form);
		my $content = $response->decoded_content;
		if(!defined($content)){
			$content=system("curl --header 'Content-Type: application/json' --request POST --data '" . encode_json(\%form)."' $url");
			print("return: $content");
			die("fail to submit benchmark data") if($content eq '');
		}
		if($response->is_success && $content=~/success/i){
			print "Submission is successful, please browse http://mcx.space/gpubench/ to see the result\n";
		}
	}
}
