#!/usr/bin/perl

use CGI;
use strict;
use DBI;
use URI::Escape;
use JSON::PP;

my ($DBName,$DBUser,$DBPass,%DBErr,$dbh,$sth,$html,$page,$jobid,$savetime,$dbname);
my $q = new CGI;
my $req;
my %jobstatus=('1'=>'queued','2'=>'starting','3'=>'running','4'=>'completed','5'=>'deleted','6'=>'invalid');

if($q->param('json') =~/"RNGSeed"/){
	$req=decode_json($q->param( 'json' ));
}

$DBName="dbi:SQLite:dbname=mcxcloud.db";
$DBUser="";
$DBPass="";
%DBErr=(RaiseError=>0,PrintError=>1);
$dbname="mcxcloud";
$savetime=time();
$jobid=sprintf("%08X%08X%08X%08X", int(rand(0x1000000000)),int(rand(0x1000000000)),int(rand(0x1000000000)),int(rand(0x1000000000)));

print "Content-Type: application/javascript\n\n";

$dbh=DBI->connect($DBName,$DBUser,$DBPass,\%DBErr) or die($DBI::errstr);

if(&V("email") ne '' && &V("json") ne ''){
    $sth=$dbh->prepare("insert into $dbname (time,name,inst,email,netname,json,shape,jobid,status,priority) values (?,?,?,?,?,?,?,?,?,?)")
                 or die "Couldn't prepare statement: ";
    $sth->execute($savetime,&V("name"),&V("inst"),&V("email"),&V("netname"),&V("json"),'',$jobid,1,50) or die($DBI::errstr);
    $html ='updatestatus({"status":"success","jobid":"'.$jobid.'","dberror":"'.$DBI::errstr.'"})'."\n";
}else{
    $jobid=&V("jobid");
    my $cmd="select status from $dbname where jobid='$jobid';";
    $sth=$dbh->selectall_arrayref("select status from $dbname where jobid='$jobid';");
    if(defined $sth->[0]){
       $html ='addlog({"status":"'.$jobstatus{join(//,@{$sth->[0]})}.'", "jobid": "'.$jobid.'"})'."\n";
    }else{
       $html ='addlog({"status":"error","jobid":"'.$jobid.'","message":"'.$cmd.'"})'."\n";
    }
}

$dbh->disconnect() or die($DBI::errstr);

print $html;

sub V{
    my ($id)=@_;
    my $val=$q->param($id);
    $val=$req->{$id} if($val eq '');
    $val=~ s/\+/ /g;
    return uri_unescape($val);
}
