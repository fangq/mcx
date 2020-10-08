#!/usr/bin/perl

use strict;
use warnings;

my $testfile='testmcx.sh';
my $mcx;
my $param=join(" ",@ARGV);
my $ldd=`which ldd`;

chomp($ldd);

if($ldd eq ''){
	$ldd="otool -L";
}

open(FTEST, '<', $testfile) or die $!;
my @lines=<FTEST>;
close(FTEST);

my %alltests=();
my %res=();

for my $ln (@lines){
	if($ln =~/^temp=`(.+)\s*\|\s*grep\s+(-o\s+-E)*\s+'(.*)'`$/){
		my $cmd=$1;
		my $pat=$3;
		$cmd=~s/\$MCX/$mcx/g;
		$cmd=~s/\$PARAM/$param/g;
		$cmd=~s/\$LDD/$ldd/g;
		$alltests{$cmd}=$pat;
	}elsif($ln =~ /^MCX=(.*)/){
		$mcx=$1;
		$mcx=~s/\$EXE/mcx/;
	}
}
foreach my $cmd (keys %alltests){
	$res{$cmd}=`$cmd`;
	if($res{$cmd}=~/$alltests{$cmd}/){
		print "Test Succeeded: $cmd\n";
		delete($res{$cmd});
	}else{
		print "Test Failed: $cmd\n";
	}
}
