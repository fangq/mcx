//
// This unit is part of the GLScene Project, http://glscene.org
//
{
    String constants that are used in many GLScene units

	 History : 
       16/09/10 - YP - Added glsUnknownParam
       23/02/07 - DaStr - Added glsDot, glsUnsupportedType, glsUncompatibleTypes,
                         glsUnknownType, glsShaderNeedsAtLeastOneLightSource(Ex),
                         glsCadencerNotDefined(Ex), glsSceneViewerNotDefined
       16/02/07 - DaStr - Added glsOCProxyObjects, glsError, glsErrorEx,
                         glsMatLibNotDefined, glsMaterialNotFoundInMatlib(Ex)
       26/08/02 - EG - Added missing header, added glsUnknownExtension
	 
}
unit GLStrings;

interface

resourcestring
  // General
  glsDot = '.';
  glsError   = 'Error!';
  glsErrorEx = 'Error: ';

  // SceneViewer
  glsNoRenderingContext = 'Could not create a rendering context';
  glsWrongVersion       = 'Need at least OpenGL version 1.1';
  glsTooManyLights      = 'Too many lights in the scene';
  glsDisplayList        = 'Failed to create a new display list for object ''%s''';
  glsWrongBitmapCanvas  = 'Couldn''t create a rendering context for the given bitmap';
  glsWrongPrinter       = 'Couldn''t render to printer';
  glsAlreadyRendering   = 'Already rendering';
  glsSceneViewerNotDefined = 'SceneViewer not defined!';

  // GLCadencer
  glsCadencerNotDefined   = 'Cadencer not defined!';
  glsCadencerNotDefinedEx = 'Cadencer not defined for  the ''%s'' component';

  // Shaders
  glsShaderNeedsAtLeastOneLightSource   = 'This shader needs at least one LightSource!';
  glsShaderNeedsAtLeastOneLightSourceEx = 'Shader ''%s'' needs at least one LightSource!';

  // GLTree
  glsSceneRoot  = 'Scene root';
  glsObjectRoot = 'Scene objects';
  glsCameraRoot = 'Cameras';
  glsCamera     = 'Camera';

  // GLTexture
  glsImageInvalid = 'Could not load texture, image is invalid';
  glsNoNewTexture = 'Could not get new texture name';

  // GLMaterials
  glsMatLibNotDefined = 'Material Library not defined!';
  glsMaterialNotFoundInMatlib = 'Material not found in current Material Library!';
  glsMaterialNotFoundInMatlibEx = 'Material "%s" not found in current Material Library!';

  // GLObjects
  glsSphereTopBottom = 'The top angle must be higher than the bottom angle';
  glsSphereStartStop = 'The start angle must be smaller than then stop angle';
  glsMaterialNotFound = 'Loading failed: could not find material %s';
  glsInterleaveNotSupported = 'Interleaved Array format not supported yet. Sorry.';

  // common messages
  glsUnknownArchive = '%s : unknown archive version %d';
  glsOutOfMemory = 'Fatal: Out of memory';
  glsFileNotFound = 'File %s not found';
  glsFailedOpenFile = 'Could not open file: %s';
  glsFailedOpenFileFromCurrentDir = 'Could not open file: %s'#13#10'(Current directory is %s)';
  glsNoDescriptionAvailable = 'No description available';
  glsUnBalancedBeginEndUpdate = 'Unbalanced Begin/EndUpdate';
  glsUnknownExtension = 'Unknown file extension (%s), maybe you forgot to add the support '
                       +'unit to your uses? (%s?)' ;
  glsMissingResource = 'Missing application resource: %s: %s';

  glsIncompatibleTypes = 'Incompatible types!';
  glsUnknownType       = 'Unknown type!';
  glsUnsupportedType   = 'Unsupported type!';

  // object categories
  glsOCBasicGeometry = 'Basic geometry';
  glsOCAdvancedGeometry = 'Advanced geometry';
  glsOCMeshObjects = 'Mesh objects';
  glsOCParticleSystems = 'Particle systems';
  glsOCEnvironmentObjects = 'Environment objects';
  glsOCSpecialObjects = 'Special objects';
  glsOCGraphPlottingObjects = 'Graph-plotting objects';
  glsOCDoodad = 'Doodad objects';
  glsOCHUDObjects = 'HUD objects';
  glsOCGuiObjects = 'GUI objects';
  glsOCProxyObjects = 'Proxy objects';
  glsOCExperimental = 'Experimental objects';

  glsUnknownParam =
    'Unknown %s "%s" for "%s" or program not in use';

  //
  // Context
  strCannotAlterAnActiveContext = 'Cannot alter an active context';
  strContextActivationFailed = 'Context activation failed: %X, %s';
  strContextAlreadyCreated = 'Context already created';
  strContextDeactivationFailed = 'Context deactivation failed';
  strContextNotCreated = 'Context not created';
  strDeleteContextFailed = 'Delete context failed';
  strFailedToShare = 'DoCreateContext - Failed to share contexts';
  strIncompatibleContexts = 'Incompatible contexts';
  strInvalidContextRegistration = 'Invalid context registration';
  strInvalidNotificationRemoval = 'Invalid notification removal';
  strNoActiveRC = 'No active rendering context';
  strUnbalancedContexActivations = 'Unbalanced context activations';
  strUnableToCreateLegacyContext = 'Unable to create legacy context';


implementation

end.

